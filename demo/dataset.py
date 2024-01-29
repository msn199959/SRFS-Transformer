import scipy.spatial
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from image import load_data_test
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
        img = load_data_test(img_path, self.args, self.train)

        img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        padding_h = img.shape[1] % self.args['crop_size']
        padding_w = img.shape[2] % self.args['crop_size']

        if padding_w != 0:
            padding_w = self.args['crop_size'] - padding_w
        if padding_h != 0:
            padding_h = self.args['crop_size'] - padding_h

        '''for padding'''
        pd = (padding_w, 0, padding_h, 0)
        img = F.pad(img, pd, 'constant')

        width, height = img.shape[2], img.shape[1]
        num_w = int(width / self.args['crop_size'])
        num_h = int(height / self.args['crop_size'])

        '''image to patch'''
        img_return = img.view(3, num_h, self.args['crop_size'], width).view(3, num_h, self.args['crop_size'], num_w,
                                                                            self.args['crop_size'])
        img_return = img_return.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, self.args['crop_size'],
                                                                            self.args['crop_size']).permute(1, 0, 2, 3)

        return fname, img_return