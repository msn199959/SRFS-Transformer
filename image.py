import cv2
import h5py
import numpy as np
from PIL import Image
import os

def load_data(img_path, args, train=True):
    if train == True:
        if args['using_refinement'] == False:
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
        else:
            if args['refine_replace'] == True:
                gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map_replace')
            else:
                if args['cur_refine_step'] == 0:
                    # 第0step依旧用原始的加载
                    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
                else:
                    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
                    gt_split = os.path.split(gt_path)
                    gt_path = '/'.join([gt_split[0]+f"_{args['cur_refine_step']}-th", gt_split[1]])
    else:
        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            img = np.asarray(gt_file['image'])
            img = Image.fromarray(img, mode='RGB')
            break
        except OSError:
            #print("path is wrong", gt_path)
            cv2.waitKey(1000)  # Wait a bit
    img = img.copy()
    k = k.copy()
    # print(f'loading a h5 file: {gt_path}')

    return img, k


def load_data_test(img_path, args, train=True):

    img = Image.open(img_path).convert('RGB')

    return img

