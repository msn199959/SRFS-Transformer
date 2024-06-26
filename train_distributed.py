from __future__ import division

import os
import warnings

from config import return_args, args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import torch.nn as nn
from torchvision import transforms
import dataset
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
import pdb
from torch.utils.tensorboard import SummaryWriter  # add tensoorboard
import shutil
from multiprocessing import cpu_count


if args.backbone == 'resnet50' or args.backbone == 'resnet101':
    from Networks.CDETR import build_model

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

def main(args):
    if args['dataset'] == 'jhu':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'nwpu':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'

    with open(train_file, 'rb') as outfile:
        train_data = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_data = np.load(outfile).tolist()

    if args['using_refinement'] and args['dataset'] == 'jhu' and args['refine_replace']:
        replace_path = f"./data/jhu_crowd_v2.0/train/gt_detr_map_replace_{args['train_number']}_2048/"
        source_path = './data/jhu_crowd_v2.0/train/gt_detr_map_2048'

        if args['local_rank'] == 0:
            if os.path.exists(replace_path):
                if args['pre'] is None :
                    shutil.rmtree(replace_path)
                    print(f'----------delete the last gt dir {replace_path} -----------')
                    shutil.copytree(source_path, replace_path)
                    print(f'-----------replace new gt dir {replace_path}-----------')
                else:
                    None
            else:
                shutil.copytree(source_path, replace_path)
                print(f'-----------replace new gt dir {replace_path}-----------')

    args['workers'] = int(cpu_count()/3)
    if args['local_rank'] == 0:
        print(f"using {int(cpu_count()/3)} threads loading dataset")

    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()

    
    if args['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args['local_rank']])
        path = './save_file/log_file/' + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        args['save_path'] = path
        if not os.path.exists(args['save_path']) and args['local_rank'] == 0:
            os.makedirs(args['save_path'])
        if args['save']:
            logger = get_root_logger(path + '/1.log')
        else:
            logger = get_root_logger('./save_file/log_file/debug/debug.log')
        if args['local_rank'] == 0:
            writer = SummaryWriter(path)

    else:
        args['train_patch'] = True
        return_args.train_patch = True
        model = nn.DataParallel(model, device_ids=[0])
        path = './save_file/log_file/debug/'
        args['save_path'] = path
        if not os.path.exists(args['save_path']):
            os.makedirs(path)
        logger = get_root_logger(path + 'debug.log')
        if args['local_rank'] == 0:
            writer = SummaryWriter(path)
    

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("model params:", num_params / 1e6)
    logger.info("model params: = {:.3f}\t".format(num_params / 1e6))

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])
    if args['local_rank'] == 0:
        logger.info(args)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['lr_step'], gamma=0.5, last_epoch=-1)

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

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
    #logger.info('best result = {:.3f}'.format(args['best_pred']))
    torch.set_num_threads(args['workers'])

    if args['local_rank'] == 0:
        logger.info('best result={:.3f}\t start epoch={:.3f}'.format(args['best_pred'], args['start_epoch']))
        logger.info('start training!')

    eval_epoch = 0
    for epoch in range(args['start_epoch'], args['epochs'] + 1):
        train(train_data, model, criterion, optimizer, epoch, scheduler, logger, writer, args)
        torch.cuda.empty_cache() #显存清理
        '''inference '''
        if epoch % args['test_per_epoch'] == 0 and epoch >= args['test_start_epoch']:
            pred_mae, pred_mse, visi = validate(test_data, model, criterion, epoch, logger, args)
            if args['local_rank'] == 0:
                writer.add_scalar('Metrcis/MAE', pred_mae, eval_epoch)
                writer.add_scalar('Metrcis/MSE', pred_mse, eval_epoch)

            # save_result
            if args['save']:
                is_best = pred_mae < args['best_pred']
                args['best_pred'] = min(pred_mae, args['best_pred'])
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args['pre'],
                    'state_dict': model.state_dict(),
                    'best_prec1': args['best_pred'],
                    'optimizer': optimizer.state_dict(),
                }, visi, is_best, args['save_path'])

            if args['local_rank'] == 0:
                logger.info(
                    'Testing Epoch:[{}/{}]\t mae={:.3f}\t mse={:.3f}\t best_mae={:.3f}\t'.format(
                        epoch,
                        args['epochs'],
                        pred_mae, pred_mse,
                        args['best_pred']))


def collate_wrapper(batch):
    targets = []
    imgs = []
    fname = []

    for item in batch:

        #if return_args.train_patch:
        fname.append(item[0])

        for i in range(0, len(item[1])):
            imgs.append(item[1][i])

        for i in range(0, len(item[2])):
            targets.append(item[2][i])
        # else:
        # fname.append(item[0])
        # imgs.append(item[1])
        # targets.append(item[2])

    return fname, torch.stack(imgs, 0), targets


def train(Pre_data, model, criterion, optimizer, epoch, scheduler, logger, writer, args):
    # losses = AverageMeter()
    torch.cuda.synchronize()
    start = time.time()

    train_data = dataset.listDataset(Pre_data, args['save_path'],
                                     shuffle=True,
                                     transform=transforms.Compose([
                                         transforms.RandomGrayscale(p=args['gray_p'] if args['gray_aug'] else 0),
                                         transforms.ToTensor(),

                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]),
                                     ]),
                                     train=True,
                                     args=args)

    if args['distributed']:
        datasampler = DistributedSampler(train_data, num_replicas=dist.get_world_size(), rank=args['local_rank'])
        datasampler.set_epoch(epoch)
    else:
        datasampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args['batch_size'],
        drop_last=False,
        collate_fn=collate_wrapper,
        sampler=datasampler,
        num_workers=args['workers'],
        prefetch_factor=2,
        pin_memory=True
    )
    
    model.train()
    loss_log = []
    '''
    # 定义一个函数来检查梯度中是否存在NaN，并在存在时打印
    def check_for_nan(grad):
        if grad is not None and torch.isnan(grad).any():
            print(f"NaN detected in the gradients of {name}.")
            print(f"Gradient: {grad}")

    # 假设您的模型是model
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 仅在需要梯度的参数上注册钩子
            param.register_hook(lambda grad, name=name: check_for_nan(grad, name))
    '''

    # criterion.weight_dict['encoder_supervise'] = max(0.5*(100-epoch)/100, 0)
    # torch.autograd.set_detect_anomaly(True)

    # import copy
    # params_before = copy.deepcopy(model.state_dict())
    
    # 记录修正点的个数
    refine_points_num = 0

    for i, (fname, img, targets) in enumerate(train_loader):
        # save_img_patch(i,img)
        # criterion.cat_target_for_test(fname, targets)
        img = img.cuda()
        d6 = model(img)
        loss_dict, record_idx_costs = criterion(d6, targets, return_idx_costs=args['using_refinement'])
        weight_dict = criterion.weight_dict
        if epoch < args['interm_start_epoch']:
            weight_dict['encoder_supervise'] = 0
        else:
            weight_dict['encoder_supervise'] = args['interm_loss_cof']
            
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # print(f"loss: {loss_dict}")
        # print(f"total_loss: {loss}")
        if args['local_rank'] == 0:
            writer.add_scalar('loss/total', loss, len(train_loader) * epoch + i)
            writer.add_scalar('loss/loss_ce', loss_dict['loss_ce'], len(train_loader) * epoch + i)
            writer.add_scalar('loss/loss_point', loss_dict['loss_point'], len(train_loader) * epoch + i)
            writer.add_scalar('lr/lr_backbone', optimizer.param_groups[0]['lr'], len(train_loader) * epoch + i)
            if args['encoder_interm_supervise']:
                writer.add_scalar('loss/loss_encoder_supervise_lr', weight_dict['encoder_supervise'], len(train_loader) * epoch + i)
                writer.add_scalar('loss/loss_encoder_supervise', loss_dict['encoder_supervise'], len(train_loader) * epoch + i)

            if args['aux_loss']:
                for i in range(5):
                    writer.add_scalar(f'loss/loss_point_{i}', loss_dict[f'loss_point_{i}'], len(train_loader) * epoch + i)
                    writer.add_scalar(f'loss/loss_ce_{i}', loss_dict[f'loss_ce_{i}'], len(train_loader) * epoch + i)

        loss_log.append(loss.item())

        optimizer.zero_grad()
        #with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()
        '''
        params_after = model.state_dict()
        # 比较参数
        for key in params_before:
            change = torch.norm(params_before[key] - params_after[key]).item()
            print(f"Parameter {key} changed by {change}")
        '''
        
        if args['using_refinement'] and epoch >= args['refine_starting_epoch'] and epoch % args['refine_interval'] == 0 and args['cur_refine_step'] < args['total_refine_step']:
            with torch.no_grad():
                refine_points_num += train_data.refine_gt(fname, d6, targets, record_idx_costs, method="high_cof_fur_distance", cof_threshold=args['cof_threshold'], distance_ratio=args['distance_ratio'])

    torch.cuda.synchronize()
    epoch_time = time.time() - start
    scheduler.step()
    if args['local_rank'] == 0:
        logger.info('Training Epoch:[{}/{}]\t loss={:.5f}\t lr={:.6f}\t epoch_time={:.3f}'.format(epoch,args['epochs'],np.mean(loss_log),args['lr'], epoch_time))
        if args['using_refinement'] and epoch >= args['refine_starting_epoch'] and epoch % args['refine_interval'] == 0 and args['cur_refine_step'] < args['total_refine_step']:
            logger.info(f"----------Refinement at {epoch} epoch and refine {refine_points_num} points--------------")
            args['cur_refine_step'] += 1

def validate(Pre_data, model, criterion, epoch, logger, args):
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

    mae = 0.0
    mse = 0.0
    visi = []

    for i, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader):

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(kpoint.shape) == 5:
            kpoint = kpoint.squeeze(0)


        with torch.no_grad():
            img = img.cuda()
            outputs = model(img)
        # import pdb;pdb.set_trace()
        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits = out_logits.view(1, -1, 2)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                               kpoint.shape[0] * args['num_queries'], dim=1)
        count = 0
        gt_count = torch.sum(kpoint).item()
        for k in range(topk_values.shape[0]):
            sub_count = topk_values[k, :]
            sub_count[sub_count < args['threshold']] = 0
            sub_count[sub_count > 0] = 1
            sub_count = torch.sum(sub_count).item()
            count += sub_count

        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)

    mae = mae / len(test_loader)
    mse = math.sqrt(mse / len(test_loader))

    print('mae', mae, 'mse', mse)
    return mae, mse, visi

def save_img_patch(fname, img):
    save_dir = './save_raw_img_patch'
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    import torchvision.transforms as transforms
    from PIL import Image
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


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
