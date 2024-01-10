import scipy.spatial
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from image import load_data
import random
from PIL import Image
import numpy as np
import h5py
import pdb

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

        self.rate = 1
        self.count = 1
        self.old_rate = []

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]
        fname = os.path.basename(img_path)
        img, kpoint = load_data(img_path, self.args, self.train)
        flip = False

        while min(kpoint.shape[0], kpoint.shape[1]) < self.args['crop_size']  and self.train == True:
            img_path = self.lines[random.randint(1, self.nSamples-1)]
            fname = os.path.basename(img_path)
            img, kpoint = load_data(img_path, self.args, self.train)

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kpoint = np.fliplr(kpoint)
                flip = True

            if self.args['scale_aug'] == True and random.random() > (1 - self.args['scale_p']): # random scale
                if self.args['scale_type'] == 0:
                    self.rate = random.choice([0.8, 0.9, 1.1, 1.2])
                elif self.args['scale_type'] == 1:
                    self.rate = random.choice([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
                elif self.args['scale_type'] == 2:
                    self.rate = random.uniform(0.7, 1.3)
                width, height = img.size
                width = int(width * self.rate)
                height = int(height * self.rate)
                if min(width, height) > self.args['crop_size']:
                    img = img.resize((width, height), Image.ANTIALIAS)
                else:
                    self.rate = 1
            else:
                self.rate = random.uniform(1.0, 1.0)

        kpoint = kpoint.copy()
        img = img.copy()

        if self.transform is not None:
            img = self.transform(img)


        if self.train == True:
            count_none = 0
            imgs = []
            targets = []
            for l in range(self.args['num_patch']):
                while True:
                    target = {}
                    if count_none > 100:
                        img_path = self.lines[random.randint(1, self.nSamples-1)]
                        fname = os.path.basename(img_path)
                        img, kpoint = load_data(img_path, self.args, self.train)
                        flip = False

                        count_none = 0
                        if self.transform is not None:
                            img = self.transform(img)
                        self.rate = 1

                    width = self.args['crop_size']
                    height = self.args['crop_size']
                    try:
                        crop_size_x = random.randint(0, img.shape[1] - width)
                        crop_size_y = random.randint(0, img.shape[2] - height)
                    except:
                        count_none = 1000
                        continue

                    '''crop image'''
                    sub_img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
                    '''crop kpoint'''
                    crop_size_x = int(crop_size_x / self.rate)
                    crop_size_y = int(crop_size_y / self.rate)
                    width = int(width / self.rate)
                    height = int(height / self.rate)
                    sub_kpoint = kpoint[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
                    # sub_kpoint[sub_kpoint != 1] = 0
                    sub_kpoint[sub_kpoint < 1] = 0
                    num_points = int(np.sum(sub_kpoint))
                    gt_points = torch.tensor(self.get_indices_with_repeats(sub_kpoint),dtype=torch.int64).view(-1, 2)
                    
                    distances = self.caculate_knn_distance(gt_points, num_points)
                    points = torch.cat([gt_points, distances], dim=1)
                    #points = gt_points

                    if num_points > self.args['min_num'] and num_points < self.args['num_queries']:
                        break

                    count_none += 1

                target['labels'] = torch.ones([1, num_points]).squeeze(0).type(torch.LongTensor)
                target['points_macher'] = torch.true_divide(points, width).type(torch.FloatTensor)
                target['points'] = torch.true_divide(points[:, 0:self.args['channel_point']], width).type(torch.FloatTensor)
                target['average_distance'] = torch.true_divide(distances, width).type(torch.FloatTensor)
                target['crop_bias'] = torch.tensor([crop_size_x, crop_size_y, width, height]).type(torch.int32)
                target['flip_aug'] = torch.as_tensor(flip, dtype=torch.bool)
                target['scale'] = torch.as_tensor(self.rate, dtype=torch.float32)

                imgs.append(sub_img)
                targets.append(target)

            return img_path, imgs, targets

        else:

            kpoint = torch.from_numpy(kpoint).cuda()

            padding_h = img.shape[1] % self.args['crop_size']
            padding_w = img.shape[2] % self.args['crop_size']

            if padding_w != 0:
                padding_w = self.args['crop_size'] - padding_w
            if padding_h != 0:
                padding_h = self.args['crop_size'] - padding_h

            '''for padding'''
            pd = (padding_w, 0, padding_h, 0)
            img = F.pad(img, pd, 'constant')
            kpoint = F.pad(kpoint, pd, 'constant').unsqueeze(0)

            width, height = img.shape[2], img.shape[1]
            num_w = int(width / self.args['crop_size'])
            num_h = int(height / self.args['crop_size'])

            '''image to patch'''
            img_return = img.view(3, num_h, self.args['crop_size'], width).view(3, num_h, self.args['crop_size'], num_w,
                                                                                self.args['crop_size'])
            img_return = img_return.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, self.args['crop_size'],
                                                                             self.args['crop_size']).permute(1, 0, 2, 3)

            kpoint_return = kpoint.view(num_h, self.args['crop_size'], width).view(num_h, self.args['crop_size'], num_w,
                                                                                   self.args['crop_size'])
            kpoint_return = kpoint_return.permute(0, 2, 1, 3).contiguous().view(num_w * num_h, 1, self.args['crop_size'],
                                                                                self.args['crop_size'])

            targets = []
            patch_info = [num_h, num_w, height, width, self.args['crop_size'], padding_w, padding_h]
            return fname, img_return, kpoint_return, targets, patch_info

    def get_indices_with_repeats(self, array):
        """
        Get the indices of non-zero elements in the array.
        For elements greater than 1, add indices multiple times (value - 1 times).

        Parameters:
        array (numpy.ndarray): The input array.

        Returns:
        list of tuples: A list containing the indices of non-zero elements.
        """
        indices = []
        for i in range(array.shape[0]):        # 遍历行
            for j in range(array.shape[1]):    # 遍历列
                if array[i, j] > 0:
                    indices.extend([[i, j]] * array[i, j])  # 根据值添加索引
        return indices

    def caculate_knn_distance(self, gt_points, num_point):

        if num_point >= 4:
            tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
            distances, locations = tree.query(gt_points, k=min(self.args['num_knn'], num_point))
            distances = np.delete(distances, 0, axis=1)
            distances = np.mean(distances, axis=1)
            distances = torch.from_numpy(distances).unsqueeze(1)

        elif num_point == 0:
            distances = gt_points.clone()[:, 0].unsqueeze(1)

        elif num_point == 1:
            tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
            distances, locations = tree.query(gt_points, k=num_point)
            distances = torch.from_numpy(distances).unsqueeze(1)

        elif num_point == 2:
            tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
            distances, locations = tree.query(gt_points, k=num_point)
            distances = np.delete(distances, 0, axis=1)
            distances = (distances[:, 0]) / 1.0
            distances = torch.from_numpy(distances).unsqueeze(1)

        elif num_point == 3:
            tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
            distances, locations = tree.query(gt_points, k=num_point)
            distances = np.delete(distances, 0, axis=1)
            distances = (distances[:, 0] + distances[:, 1]) / 2
            distances = torch.from_numpy(distances).unsqueeze(1)

        return distances
    
    def refine_gt(self, img_paths, outputs, targets, record_idx_costs):
        '''
        refine the latest gt(.h5) file, and save a newer(the refine_step)
        input:
        img_path: record img path
        record_idx_costs: record out_idx, tgt_idx and cost
        gt_data_dir: dir of saving gt data
        '''
        ## 需要注意存在 flip aug等增强操作

        '''
         outputs.keys()
        dict_keys(['pred_logits', 'pred_points', 'aux_outputs', 'interm_outputs', 'interm_outputs_for_matching_pre', 'dn_meta'])
        '''

        assert 'pred_logits' in outputs and 'pred_points' in outputs
        assert len(img_paths) > 0

        outputs_logits_cords = {}
        for k, v in outputs.items():
            if k == 'pred_points' or k == 'pred_logits':
                outputs_logits_cords[k] = v.to('cpu')
            else:
                continue
        ### 保存
        # img_paths = './data/jhu_crowd_v2.0/val/images_2048/4340.jpg'
        gt_data_dir = os.path.split(img_paths[0])[0].replace('images', 'gt_detr_map')
        if self.args['refine_replace'] == True:
            last_gt_data_dir = os.path.split(img_paths[0])[0].replace('images', f"gt_detr_map_replace_{self.args['train_number']}")
            new_gt_data_dir = last_gt_data_dir
            assert os.path.exists(new_gt_data_dir)
        else:
            if self.args['cur_refine_step'] == 0:
                last_gt_data_dir = gt_data_dir
                new_gt_data_dir = f'{gt_data_dir}_{1}-th'
            else:
                last_gt_data_dir = f"{gt_data_dir}_{self.args['cur_refine_step']}-th"
                new_gt_data_dir = f"{gt_data_dir}_{self.args['cur_refine_step']+1}-th"

            if not os.path.exists(new_gt_data_dir):
                    os.makedirs(new_gt_data_dir)

        for i, (img_path, output_confs, output_cords, target, record_idx_cost) in enumerate(zip(img_paths, outputs_logits_cords['pred_logits'], outputs_logits_cords['pred_points'][...,:2], targets, record_idx_costs)):
            kpoint = self.load_refined_data(img_path)
            target = {k: v.to('cpu') for k, v in target.items()}
            crop_size_x, crop_size_y, width, height = target['crop_bias']
            if target['flip_aug']:
                kpoint = np.fliplr(kpoint)

            sub_latest_kpoint = kpoint[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            target_points_idx = record_idx_cost[:, 1].long()
            
            ### 这个主要是先检查
            training_points = torch.tensor(self.get_indices_with_repeats(sub_latest_kpoint),dtype=torch.int64).view(-1, 2)[target_points_idx]
            '''
            # type_1 test
            training_points = torch.as_tensor(torch.round(target['points'][...,:2][target_points_idx]*width), dtype=torch.int32) #这里会引入误差
            gt_points = torch.tensor(self.get_indices_with_repeats(sub_latest_kpoint),dtype=torch.int64).view(-1, 2)[target_points_idx]
            assert torch.all(training_points == gt_points)

            # type_2 test 这种容易有问题
            reconstructed_mask = np.zeros((height, width), dtype=int)
            training_points_np = training_points.numpy()
            np.add.at(reconstructed_mask, (training_points_np[:, 0], training_points_np[:, 1]), 1)

            try:
                assert np.all(reconstructed_mask == sub_latest_kpoint)
            except AssertionError as e:
                print("AssertionError:", e)
                pdb.set_trace()
            '''

            predicted_idx = record_idx_cost[:, 0].long()
            predicted_points = torch.as_tensor(output_cords[predicted_idx]*width, dtype=torch.int32)
            predicted_logits = torch.as_tensor(output_confs[predicted_idx].sigmoid(), dtype=torch.float32)

            cost = record_idx_cost[:, 2]
            cost_sort_idx = torch.sort(cost).indices

            sorted_predict_points = predicted_points[cost_sort_idx]
            sorted_training_points = training_points[cost_sort_idx]
            sorted_predict_logits = predicted_logits[cost_sort_idx][:, -1].unsqueeze(1)

            pred_cof = self.args['refine_weight'] * sorted_predict_logits
            gt_cof = 1 - pred_cof
            refined_gt_points = (torch.round(gt_cof*sorted_training_points + pred_cof*sorted_predict_points).to(torch.int32)).numpy()

            refined_results = np.zeros((height, width), dtype=int)
            np.add.at(refined_results, (refined_gt_points[:, 0], refined_gt_points[:, 1]), 1)

            kpoint[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height] = refined_results

            if target['flip_aug']:
                kpoint = np.fliplr(kpoint)
            
            self.save_refined_data(img_path, last_gt_data_dir, new_gt_data_dir, kpoint)


    def load_refined_data(self, img_path):
        if self.args['using_refinement'] == False:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
        else:
            if self.args['refine_replace'] == True:
                gt_path = img_path.replace('.jpg', '.h5').replace('images', f"gt_detr_map_replace_{self.args['train_number']}")
            else:
                if self.args['cur_refine_step'] == 0:
                    # 第0step依旧用原始的加载
                    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
                else:
                    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
                    gt_split = os.path.split(gt_path)
                    gt_path = '/'.join([gt_split[0]+f"_{self.args['cur_refine_step']}-th", gt_split[1]])
        while True:
            try:
                gt_file = h5py.File(gt_path)
                k = np.asarray(gt_file['kpoint'])
                break
            except OSError:
                break
        k = k.copy()
        return k
    
    def save_refined_data(self, img_path, last_dir, output_dir, modified_kpoint_data):
        '''
        img_path: record h5 idx
        last_dir: load the last gt
        output_dir: save the new gt
        point_content: 
        '''
        gt_postfix = os.path.split(img_path)[-1].replace('.jpg', '.h5')

        original_file_path = os.path.join(last_dir, gt_postfix)
        new_file_path = os.path.join(output_dir, gt_postfix)

        assert os.path.exists(original_file_path) and os.path.exists(output_dir)

        if self.args['refine_replace']==True:
            with h5py.File(new_file_path, 'r+') as file:
                # 假设'kpoint'是文件中的一个数据集
                if 'kpoint' in file:
                    file['kpoint'][...] = modified_kpoint_data
                else:
                    print("'kpoint' not found in the file")
        else:
            with h5py.File(original_file_path, 'r') as original_file:
                # 创建新的 H5 文件，准备将修改后的内容写入其中
                with h5py.File(new_file_path, 'w') as new_file:
                    # 复制原始文件的组结构和数据集（除了'kpoint'部分）
                    for item_name in original_file:
                        if item_name != 'kpoint':
                            original_item = original_file[item_name]
                            new_file.copy(original_item, new_file)

                    # 将修改后的'kpoint'数据写入新文件
                    new_file.create_dataset('kpoint', data=modified_kpoint_data)

        assert os.path.exists(new_file_path)