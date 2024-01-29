from __future__ import division

import os
import warnings
import torch
from config import return_args, args
torch.cuda.set_device(int(args.gpu_id[0]))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import torch.nn as nn
from torchvision import transforms
from . import dataset
import math
from utils import get_root_logger, setup_seed
import nni
from nni.utils import merge_parameter
import time
import util.misc as utils
from utils import save_checkpoint
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # add tensoorboard

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

if args.backbone == 'resnet50' or args.backbone == 'resnet101':
    from Networks.CDETR import build_model

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

def main(args):
    if args['dataset'] == 'jhu':
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'nwpu':
        test_file = './npydata/nwpu_val.npy'

    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)

    model = model.cuda()

    model = nn.DataParallel(model, device_ids=[int(data) for data in list(args['gpu_id']) if data!=','])
    path = './save_file/log_file/debug/'
    args['save_path'] = path
    if not os.path.exists(args['save_path']):
        os.makedirs(path)
    logger = get_root_logger(path + 'debug.log')
    writer = SummaryWriter(path)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("model params:", num_params / 1e6)
    logger.info("model params: = {:.3f}\t".format(num_params / 1e6))

    if args['local_rank'] == 0:
        logger.info(args)


    if args['pre']:
        if os.path.isfile(args['pre']):
            logger.info("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            logger.info("=> no checkpoint found at '{}'".format(args['pre']))

    print('best result:', args['best_pred'])
    logger.info('best result = {:.3f}'.format(args['best_pred']))
    torch.set_num_threads(args['workers'])

    if args['local_rank'] == 0:
        logger.info('best result={:.3f}\t start epoch={:.3f}'.format(args['best_pred'], args['start_epoch']))

    test_data = test_list
    if args['local_rank'] == 0:
        logger.info('start training!')

    validate(test_data, model, criterion, logger, args)


def collate_wrapper(batch):
    targets = []
    imgs = []
    fname = []

    for item in batch:

        if return_args.train_patch:
            fname.append(item[0])

            for i in range(0, len(item[1])):
                imgs.append(item[1][i])

            for i in range(0, len(item[2])):
                targets.append(item[2][i])
        else:
            fname.append(item[0])
            imgs.append(item[1])
            targets.append(item[2])

    return fname, torch.stack(imgs, 0), targets


def validate(Pre_data, model, criterion, logger, args):
    if args['local_rank'] == 0:
        logger.info('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            args=args, train=False),
        batch_size=1,
    )

    model.eval()

    for i, (fname, img) in enumerate(test_loader):

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        save_img_patch(i, img)

        with torch.no_grad():
            img = img.cuda()
            outputs = model(img)

        interm_density = outputs['intermediate_memory']

        visualize_and_save_feature_maps_as_images(interm_density)
        targets_squeezed = []
        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits = out_logits.view(1, -1, 2)



def min_max_norm(input_tensor):
    # Min and max values calculated over the last two dimensions (for each 8x8 tensor)
    tensor_min = input_tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    tensor_max = input_tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    # Prevent division by zero
    delta = tensor_max - tensor_min
    delta[delta == 0] = 1

    # Perform min-max scaling normalization
    tensor_normalized = (input_tensor - tensor_min) / delta
    return tensor_normalized

def sum_region(kpoint):
    batch_size = kpoint.shape[0]

    # Reshaping and summing to get the desired output of size [batch_size, 1, 8, 8]
    output_tensor_binary = kpoint.reshape(batch_size, 1, 8, 32, 8, 32).sum(dim=[3, 5])

    return output_tensor_binary

def save_img_patch(fname, img):
    save_dir = './save_raw_img_patch'
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    
    # 可能需要的反标准化步骤
    # 如果您的图像已经是 [0, 1] 范围内，则跳过此步骤
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],std=[1/s for s in std])

    # 准备转换操作
    to_pil = transforms.ToPILImage()

    img_name = fname+1

    # 遍历并保存每个图像
    for i in range(img.shape[0]):
        # 反标准化
        img_normalized = inv_normalize(img[i])
        # 转换为 PIL 图像
        img_pil = to_pil(img_normalized)
        # 保存图像
        img_pil.save(os.path.join(save_dir, f'{img_name}_patch_{i}.png'))


def visualize_and_save_feature_maps_as_images(feature_maps, img_id, layer_num):
        """
        Visualize and save the feature maps as images.

        :param feature_maps: A tensor of shape [H*W, batch_size, 256]
        :param output_folder: Folder where images will be saved
        """
        output_folder = './save_feature_map'
        if os.path.exists(output_folder) == False:
            os.mkdir(output_folder)
        # Assuming feature_maps is a PyTorch tensor
        H_W, batch_size, _ = feature_maps.shape
        H = int(np.sqrt(H_W))  # Assuming the feature map forms a square

        for i in range(batch_size):
            # Extract the feature map for the i-th sample in the batch
            sample_feature_map = feature_maps[:, i, :]

            # Reshape and scale the feature map to [0, 1] for visualization
            image = sample_feature_map.view(H, H, -1).mean(dim=2).cpu().numpy()
            image = (image - image.min()) / (image.max() - image.min())

            # Save the image
            plt.imshow(image, cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.savefig(f"{output_folder}/{img_id}_fm_layer_{layer_num}_sample_{i}.png")
            plt.close()


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
