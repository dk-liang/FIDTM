# coding: utf-8

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter

import torch
import cv2

# set the root to the JHU dataset you download
root = './jhu_crowd_v2.0'

train = root + '/train/images/'
val = root + '/val/images/'
test = root + '/test/images/'

'''mkdir directories'''
if not os.path.exists(train.replace('images', 'images_2048')):
    os.makedirs(train.replace('images', 'images_2048'))
if not os.path.exists(train.replace('images', 'gt_rdt_map_2048')):
    os.makedirs(train.replace('images', 'gt_rdt_map_2048'))
if not os.path.exists(train.replace('images', 'gt_show')):
    os.makedirs(train.replace('images', 'gt_show'))

if not os.path.exists(val.replace('images', 'images_2048')):
    os.makedirs(val.replace('images', 'images_2048'))
if not os.path.exists(val.replace('images', 'gt_rdt_map_2048')):
    os.makedirs(val.replace('images', 'gt_rdt_map_2048'))
if not os.path.exists(val.replace('images', 'gt_show')):
    os.makedirs(val.replace('images', 'gt_show'))

if not os.path.exists(test.replace('images', 'images_2048')):
    os.makedirs(test.replace('images', 'images_2048'))
if not os.path.exists(test.replace('images', 'gt_rdt_map_2048')):
    os.makedirs(test.replace('images', 'gt_rdt_map_2048'))
if not os.path.exists(test.replace('images', 'gt_show')):
    os.makedirs(test.replace('images', 'gt_show'))

path_sets = [train, val, test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

for img_path in img_paths:

    img = cv2.imread(img_path)
    print(img_path)
    rate = 1
    if img.shape[1] >= img.shape[0] and img.shape[1] >= 2048:
        rate = 2048.0 / img.shape[1]
    elif img.shape[0] >= img.shape[1] and img.shape[0] >= 2048:
        rate = 2048.0 / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=rate, fy=rate)

    k = np.zeros((img.shape[0], img.shape[1]))
    d_map = (np.zeros([img.shape[0], img.shape[1]]) + 255).astype(np.uint8)
    gt_file = np.loadtxt(img_path.replace('images', 'gt').replace('jpg', 'txt'))
    fname = img_path.split('/')[-1]

    try:
        y = gt_file[:, 0] * rate
        x = gt_file[:, 1] * rate
        for i in range(0, len(x)):
            if int(x[i]) < img.shape[0] and int(y[i]) < img.shape[1]:
                k[int(x[i]), int(y[i])] += 1
                d_map[int(x[i]), int(y[i])] = d_map[int(x[i]), int(y[i])] - 255
    except Exception:
        try:
            y = gt_file[0] * rate
            x = gt_file[1] * rate

            for i in range(0, 1):
                if int(x) < img.shape[0] and int(y) < img.shape[1]:
                    k[int(x), int(y)] += 1
                    d_map[int(x[i]), int(y[i])] = d_map[int(x[i]), int(y[i])] - 255
        except Exception:
            ''' this image without person'''
            k = np.zeros((img.shape[0], img.shape[1]))

    kpoint = k.copy()

    '''rdt map '''
    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 5)
    rdt_map = 1 / (distance_map + 1)
    if np.sum(kpoint) == 0:
        rdt_map = rdt_map * 0

    with h5py.File(img_path.replace('images', 'gt_rdt_map_2048').replace('jpg', 'h5'), 'w') as hf:
        hf['kpoint'] = kpoint
        hf['rdt_map'] = rdt_map

    cv2.imwrite(img_path.replace('images', 'images_2048'), img)

    '''visual'''
    rdt_map = rdt_map
    rdt_map = rdt_map / np.max(rdt_map) * 255
    rdt_map = rdt_map.astype(np.uint8)
    rdt_map = cv2.applyColorMap(rdt_map, 2)

    gt_show = img_path.replace('images', 'gt_show')
    cv2.imwrite(gt_show, rdt_map)

print("end")
