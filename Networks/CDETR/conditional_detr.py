# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math

import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
import pdb
import numpy as np
import scipy.stats
from dm_count.bregman_pytorch import sinkhorn
from dm_count.ot_loss import OT_Loss


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, channel_point, aux_loss=False, encoder_interm_supervise=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.point_embed = MLP(hidden_dim, hidden_dim, channel_point, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init point_mebed
        nn.init.constant_(self.point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].bias.data, 0)

        self.encoder_interm_supervise = encoder_interm_supervise

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_points": The normalized points coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        hs, reference, intermediate_memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        reference_before_sigmoid = inverse_sigmoid(reference) #[128, num_queries, 2]
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.point_embed(hs[lvl]) #[batch_size, num_queries, 3]
            tmp[..., :2] += reference_before_sigmoid
            #tmp[..., :2].shape = [batch_size, num_queries, 2] 
            #reference_before_sigmoid.shape = [batch_size, num_queries, 2] 
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.aux_loss: #true
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.encoder_interm_supervise:
            out['intermediate_memory'] = intermediate_memory
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_points': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth points and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, encoder_interm_supervise=False, num_interm_features=2, ot_loss=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.encoder_interm_supervise = encoder_interm_supervise
        self.DM_OT_loss = ot_loss
        self.num_interm_features = num_interm_features

    def loss_labels(self, outputs, targets, indices, num_points, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).cuda()
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_points, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_points):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty points
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_points(self, outputs, targets, indices, num_points):
        """Compute the losses related to the bounding points, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 4]
           The target points are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_points' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0).cuda()
        loss_point = F.l1_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_point.sum() / num_points

        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_points),
        #     box_ops.box_cxcywh_to_xyxy(target_points)))
        # losses['loss_giou'] = loss_giou.sum() / num_points
        # losses['loss_giou'] = 0.0
        return losses

    def wasserstein_distance_1d(self, u_values, v_values):
        """计算一维分布之间的Wasserstein距离"""
        u_sorted, _ = torch.sort(u_values)
        v_sorted, _ = torch.sort(v_values)
        return torch.mean(torch.abs(u_sorted - v_sorted))
    
    def target_map_to_grid(self, targets, grid_size=8):
        # 创建一个 3x3 的零矩阵
        grid = torch.zeros((grid_size, grid_size), dtype=torch.int)

        # 遍历每个点
        for point in targets:
            x, y, _ = point  # 忽略 distance，只考虑 x 和 y

            # 确定点落在哪个网格单元
            # 由于 x, y 在 0-1 之间，乘以 grid_size 并取整即可找到对应的索引
            x_index = int(x * grid_size)
            y_index = int(y * grid_size)

            # 防止坐标正好在边界上（例如 1.0）导致的索引越界
            x_index = min(x_index, grid_size - 1)
            y_index = min(y_index, grid_size - 1)

            # 增加对应网格单元的值
            grid[x_index, y_index] += 1

        return grid
    
    def loss_encoder_supervise(self, u_values, v_values):
        # u_values = intermediate_memory [2, b, 64, 256] 256=channels
        # v_values = target
        u_values = torch.mean(u_values, dim=3, keepdim=True).squeeze() #[2, batch_size, 64, 256]->[2, batch_size, 64] 
        _, batch_size, _ = u_values.shape 
        v_values = [self.target_map_to_grid(v_value['points']) for v_value in v_values]
        v_values = torch.stack(v_values).transpose(1,2).cuda()
        v_values = v_values.reshape(batch_size,-1) #[batch_size, 64]

        v_values_normlized = 1 - self.min_max_norm(v_values) #[batch_size, 64]
        u_values_normlized = self.min_max_norm(u_values) #[2,batch_size, 64]

        v_values_normlized = v_values_normlized.unsqueeze(0).repeat(2,1,1) #[2, batch_size, 64]

        losses = {'encoder_supervise': torch.mean(torch.abs(u_values_normlized - v_values_normlized))}
        return losses
    
    def loss_encoder_supervise_OT(self, u_values, v_values):
        # u_values = intermediate_memory [2, b, 64, 256] 256=channels
        # v_values = target
        u_values = torch.mean(u_values, dim=3, keepdim=True).squeeze() #[2,batch_size, 64] 
        _, batch_size, _ = u_values.shape 
        v_values = [self.target_map_to_grid(v_value['points']) for v_value in v_values]
        v_values = torch.stack(v_values).transpose(1,2).cuda()
        v_values = v_values.reshape(batch_size,-1) #[batch_size, 64]
        
        # 生成网格以表示每个元素的坐标
        N = int(np.sqrt(v_values.shape[-1]))
        coordinates = torch.tensor([[i, j] for i in range(N) for j in range(N)], dtype=torch.float32)
        # 计算所有点对之间的空间位置 L2 距离
        cost_matrix_optimized = torch.cdist(coordinates, coordinates, p=2).cuda()

        u_values = u_values.permute(1, 0, 2)
        u_values_normlized = 1 - self.min_max_norm(u_values) #[2,batch_size, 64]
        u_values_softmax = F.softmax(u_values_normlized, dim=-1) #[batch_size, 2, 64]

        v_values_float = v_values.to(dtype=torch.float32)
        v_values_softmax = self.softmax_on_nonzero_with_fallback(v_values_float)
        v_values_softmax = v_values_softmax.unsqueeze(1).expand(-1, 2, -1) # [batch_size, 2, 64]

        # losses = {'encoder_supervise': torch.mean(torch.abs(u_values_softmax - v_values_softmax))}
        losses = {'encoder_supervise': self.sinkhorn_iterations_batched(u_values_softmax, v_values_softmax, cost_matrix_optimized)}
        return losses
    
    def loss_encoder_supervise_DM(self, interm_density, targets):
        """之前预想是对最后两层做supervise， 现在可以只对最后一层做，因为有位置编码的干扰"""
        # interm_density = intermediate_memory [2, b, 64, 256] 256=channels
        # target_points = target

        interm_density = torch.mean(interm_density, dim=3, keepdim=True).squeeze() #[2,batch_size, 64] 
        points = [target['points'][:, :2].to(interm_density.device) for target in targets]
        # v_values = [self.target_map_to_grid(v_value['points']) for v_value in targets]

        interm_density = 1 - self.min_max_norm(interm_density) #[2,batch_size, 64]

        N = int(np.sqrt(interm_density.shape[-1]))
        channels, batch_size, HW = interm_density.shape
        interm_density_reshape = interm_density.reshape(channels, batch_size, 1, N, N)
        total_ot_loss = 0

        for i in range(interm_density_reshape.shape[0]):
            sub_interm_density = interm_density_reshape[i, ...]
            interm_density_norm = self.DM_norm(sub_interm_density)
            ot_loss, wd, ot_obj_value = self.DM_OT_loss(interm_density_norm, sub_interm_density, points)
            total_ot_loss = total_ot_loss + ot_loss

        losses = {'encoder_supervise': total_ot_loss}
        return losses
    

    def loss_encoder_supervise_TV(self, interm_density, targets):
        tv_loss = nn.L1Loss(reduction='none')

        N = int(np.sqrt(interm_density.shape[-2]))
        discrete_map_list = [self.target_map_to_grid(v_value['points'], N) for v_value in targets]
        discrete_map = torch.stack(discrete_map_list).unsqueeze(1).cuda() # shape = [batch, 8, 8]

        # u_values = intermediate_memory [2, b, 64, 256] 256=channels
        # v_values = target
        if self.num_interm_features == 1:
            interm_density = interm_density[0]
            interm_density = interm_density.unsqueeze(0)
            assert len(interm_density.shape) == 4
        interm_density_mean = torch.mean(interm_density, dim=3, keepdim=True).squeeze() #[2,batch_size, 64] 
        channel, batch_size, hw = interm_density_mean.shape
        interm_density_mean = interm_density_mean.reshape(batch_size, channel, N, N)
        interm_density_relu = torch.relu(interm_density_mean)
        interm_density_norm = self.DM_norm(interm_density_relu)

        # gd_count = np.array([len(p['points'].shape(0)) for p in targets], dtype=np.float32)
        gd_count = np.array([p['points'].shape[0] for p in targets], dtype=np.float32)
        gd_count_tensor = torch.from_numpy(gd_count).float().unsqueeze(1).unsqueeze(2).unsqueeze(3).cuda()
        gt_discrete_normed = discrete_map / (gd_count_tensor + 1e-6)

        total_tv_loss = tv_loss(interm_density_norm, gt_discrete_normed).sum(1).sum(1).sum(1) * (torch.from_numpy(gd_count).float().cuda()).mean(0)

        losses = {'encoder_supervise': torch.sum(total_tv_loss)}
        return losses

    def gen_discrete_map(self, im_height, im_width, points):
        """
            func: generate the discrete map.
            points: [num_gt, 2], for each row: [width, height]
            """
        discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
        h, w = discrete_map.shape[:2]
        num_gt = points.shape[0]
        if num_gt == 0:
            return discrete_map
        
        # fast create discrete map
        points_np = np.array(points).round().astype(int)
        p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
        p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
        p_index = torch.from_numpy(p_h* im_width + p_w)
        discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).numpy()

        ''' slow method
        for p in points:
            p = np.round(p).astype(int)
            p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
            discrete_map[p[0], p[1]] += 1
        '''
        assert np.sum(discrete_map) == num_gt
        return discrete_map
    
    def loss_encoder_supervise_DM(self, interm_density, targets):
        """之前预想是对最后两层做supervise， 现在可以只对最后一层做，因为有位置编码的干扰"""
        # interm_density = intermediate_memory [2, b, 64, 256] 256=channels
        # target_points = target

        interm_density = torch.mean(interm_density, dim=3, keepdim=True).squeeze() #[2,batch_size, 64] 
        points = [target['points'][:, :2].to(interm_density.device) for target in targets]
        # v_values = [self.target_map_to_grid(v_value['points']) for v_value in targets]

        interm_density = 1 - self.min_max_norm(interm_density) #[2,batch_size, 64]

        N = int(np.sqrt(interm_density.shape[-1]))
        channels, batch_size, HW = interm_density.shape
        interm_density_reshape = interm_density.reshape(channels, batch_size, 1, N, N)
        total_ot_loss = 0

        for i in range(interm_density_reshape.shape[0]):
            sub_interm_density = interm_density_reshape[i, ...]
            interm_density_norm = self.DM_norm(sub_interm_density)
            ot_loss, wd, ot_obj_value = self.DM_OT_loss(interm_density_norm, sub_interm_density, points)
            total_ot_loss = total_ot_loss + ot_loss

        losses = {'encoder_supervise': total_ot_loss}
        return losses
    
    def loss_encoder_supervise_DM_for_test(self, fname, interm_density, targets):
        # interm_density = intermediate_memory [2, b, 64, 256] 256=channels
        # target_points = target
        interm_density_mean = torch.mean(interm_density, dim=3, keepdim=True).squeeze().to(interm_density.device) #[2,batch_size, 64] 
        points = [target['points'][:, :2].to(interm_density.device) for target in targets]
        v_values = [self.target_map_to_grid(v_value['points']) for v_value in targets]

        interm_density_norm = self.min_max_norm(interm_density_mean)
        interm_density_flip = 1 - interm_density_norm #[2,batch_size, 64]

        N = int(np.sqrt(interm_density_flip.shape[-1]))
        channels, batch_size, HW = interm_density_flip.shape
        interm_density_reshape = interm_density_flip.reshape(channels, batch_size, 1, N, N)
        total_ot_loss = 0

        for i in range(interm_density_reshape.shape[0]):
            sub_interm_density = interm_density_reshape[i, ...]
            sub_interm_density_norm = self.DM_norm(sub_interm_density)
            ot_loss, wd, ot_obj_value = self.DM_OT_loss(sub_interm_density_norm, sub_interm_density, points)
            total_ot_loss = total_ot_loss + ot_loss

        losses = {'encoder_supervise': total_ot_loss}
        return losses
    
    def cat_target_for_test(self, fname, targets):
        # interm_density = intermediate_memory [2, b, 64, 256] 256=channels
        # target_points = target
        import pdb; pdb.set_trace()
        v_values = [self.target_map_to_grid(v_value['points']) for v_value in targets]
        print(1)
    
    def DM_norm(self, mu):
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu_normed

    def softmax_on_nonzero_with_fallback(self, tensor):
        # 创建一个掩码，标记非零元素
        mask = tensor != 0
        # 初始化一个全零的输出tensor
        output_tensor = torch.zeros_like(tensor)

        # 对每个batch内的非零元素应用softmax
        for i in range(tensor.size(0)):
            # 提取非零元素
            non_zero_elements = tensor[i][mask[i]]
            # 检查是否有非零元素
            if len(non_zero_elements) > 0:
                # 对非零元素应用softmax
                softmax_non_zero_elements = F.softmax(non_zero_elements, dim=0)
                # 将softmax结果放回原位置
                output_tensor[i][mask[i]] = softmax_non_zero_elements
            else:
                # 如果所有元素都是零，则将每个元素设置为1/shape[1]
                output_tensor[i] = 1.0 / tensor.size(1)
        return output_tensor
    
    def sinkhorn_iterations_batched(self, a, b, M, reg=0.1, num_iters=100):
        """
        Perform Sinkhorn iterations for batched inputs with an additional dimension.
        
        :param a: Batched source histograms (tensor of shape [batch_size, 2, n*n]).
        :param b: Batched target histograms (tensor of shape [batch_size, 2, n*n]).
        :param M: Cost matrix.
        :param reg: Regularization term.
        :param num_iters: Number of iterations.
        :return: Total loss calculated over all batches and transport matrices.
        """
        batch_size, channels = a.shape[0], a.shape[1]
        total_loss = 0
        device = a.device

        # Assuming the third dimension of a and b is n*n
        n_square = a.shape[2]
        transport_matrices = torch.zeros(batch_size, 2, n_square, n_square).to(device)

        for i in range(batch_size):
            for j in range(channels):
                a_batch = a[i, j, :].squeeze()
                b_batch = b[i, j, :].squeeze()
                K = torch.exp(-M / reg)
                Kp = (1 / a_batch).unsqueeze(1) * K

                u = torch.ones_like(a_batch)
                for _ in range(num_iters):
                    v = b_batch / (K.T @ u)
                    u = 1 / (Kp @ v)

                transport_matrix = u.unsqueeze(1) * K * v.unsqueeze(0)
                transport_matrices[i, j, :, :] = transport_matrix

                # Calculate the loss for this pair (e.g., as the Frobenius norm of the transport matrix)
                # batch_loss = torch.norm(transport_matrix, p='fro') 这是对转移矩阵求范式作为损失
                batch_loss = torch.sum(transport_matrix * M) # 这是转移乘cost
                total_loss += batch_loss

        total_loss = total_loss/batch_size
        return total_loss
    
    def min_max_norm(self, input_tensor):
        tensor_min = input_tensor.min(dim=-1, keepdim=True)[0]
        tensor_max = input_tensor.max(dim=-1, keepdim=True)[0]

        # 防止分母为零的情况
        delta = tensor_max - tensor_min
        # delta[delta == 0] = 1

        # 执行最小-最大缩放归一化
        tensor_normalized = (input_tensor - tensor_min) / delta
        return tensor_normalized
    
    def wasserstein_distance_tensor_global(self, tensor1, tensor2):
        if not tensor1.is_cuda or not tensor2.is_cuda:
            raise ValueError("Tensors must be on GPU.")

        # 将张量转换为一维
        flat_tensor1 = tensor1.view(-1)
        flat_tensor2 = tensor2.view(-1)

        # 计算Wasserstein距离
        wd = self.wasserstein_distance_1d(flat_tensor1, flat_tensor2)

        return wd

    def loss_masks(self, outputs, targets, indices, num_points):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_points, h, w]
        """
        assert "pred_masks" in outputs  #没有计算这个loss
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_points),
            "loss_dice": dice_loss(src_masks, target_masks, num_points),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'points': self.loss_points,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, return_idx_costs=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices, results_idx_costs = self.matcher(outputs_without_aux, targets, record_flag=return_idx_costs)
        # Compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # Compute all the requested losses

        #outputs['pred_points'].shape = [batch_size, queries_num, 3]
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points))
        
        if self.encoder_interm_supervise:
            # losses.update(self.loss_encoder_supervise_OT(outputs['intermediate_memory'], targets))
            # losses.update(self.loss_encoder_supervise_DM(outputs['intermediate_memory'], targets))
            losses.update(self.loss_encoder_supervise_TV(outputs['intermediate_memory'], targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_points, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, results_idx_costs


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_points = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        points = box_ops.box_cxcywh_to_xyxy(out_point)
        points = torch.gather(points, 1, topk_points.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        points = points * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'points': b} for s, l, b in zip(scores, labels, points)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 2 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = ConditionalDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        channel_point = args.channel_point,
        aux_loss=args.aux_loss,
        encoder_interm_supervise=args.encoder_interm_supervise
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_point': args.point_loss_coef}

    ot_loss = None
    if args.encoder_interm_supervise:
        weight_dict.update({'encoder_supervise': args.interm_loss_cof})
        #downsample_ratio = 32
        #ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, device, args.num_of_iter_in_ot, args.reg)

    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'points', 'cardinality']
    if args.masks:
        losses += ["masks"]
    
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             encoder_interm_supervise=args.encoder_interm_supervise,
                             num_interm_features=args.num_interm_features)
    criterion.to(device)
    postprocessors = {'point': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
