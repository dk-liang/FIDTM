import torch
from torch.utils.data import Dataset
import os
import random
from image import *
import numpy as np
import numbers
from torchvision import datasets, transforms

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

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        if self.args['preload'] == True:
            fname = self.lines[index]['fname']
            img = self.lines[index]['img']
            kpoint = self.lines[index]['kpoint']
            rdt_map = self.lines[index]['rdt_map']
            mask_map = self.lines[index]['mask_map']

        else:
            img_path = self.lines[index]
            fname = os.path.basename(img_path)
            img, rdt_map, kpoint,mask_map = load_data_fidt(img_path, self.args, self.train)

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                rdt_map = np.fliplr(rdt_map)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_map = np.fliplr(mask_map)
                kpoint = np.fliplr(kpoint)


            if random.random() > self.args['random_noise']:
                proportion = random.uniform(0.004, 0.015)
                width, height = img.size[0], img.size[1]
                num = int(height * width * proportion)
                for i in range(num):
                    w = random.randint(0, width - 1)
                    h = random.randint(0, height - 1)
                    if random.randint(0, 1) == 0:
                        img.putpixel((w, h), (0, 0, 0))
                    else:
                        img.putpixel((w, h), (255, 255, 255))

        rdt_map = rdt_map.copy()
        kpoint = kpoint.copy()
        img = img.copy()
        mask_map = mask_map.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.train == True:
            rdt_map = torch.from_numpy(rdt_map).cuda()

            width = 256
            height = 256
            # print(img.shape)
            crop_size_x = random.randint(0, img.shape[1] - width)
            crop_size_y = random.randint(0, img.shape[2] - height)
            img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            kpoint = kpoint[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            rdt_map = rdt_map[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]
            mask_map = mask_map[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height]


        return fname, img, rdt_map, kpoint, mask_map

