import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter

'''change your path'''
root = '/home/dkliang/projects/synchronous/dataset/ShanghaiTech/'

part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')

path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]

if not os.path.exists(part_A_train.replace('images', 'gt_fidt_map')):
    os.makedirs(part_A_train.replace('images', 'gt_fidt_map'))

if not os.path.exists(part_A_test.replace('images', 'gt_fidt_map')):
    os.makedirs(part_A_test.replace('images', 'gt_fidt_map'))

if not os.path.exists(part_A_train.replace('images', 'gt_show')):
    os.makedirs(part_A_train.replace('images', 'gt_show'))

if not os.path.exists(part_A_test.replace('images', 'gt_show')):
    os.makedirs(part_A_test.replace('images', 'gt_show'))

if not os.path.exists(part_B_train.replace('images', 'gt_fidt_map')):
    os.makedirs(part_B_train.replace('images', 'gt_fidt_map'))

if not os.path.exists(part_B_test.replace('images', 'gt_fidt_map')):
    os.makedirs(part_B_test.replace('images', 'gt_fidt_map'))

if not os.path.exists(part_B_train.replace('images', 'gt_show')):
    os.makedirs(part_B_train.replace('images', 'gt_show'))

if not os.path.exists(part_B_test.replace('images', 'gt_show')):
    os.makedirs(part_B_test.replace('images', 'gt_show'))

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()


def rdt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map


for img_path in img_paths:
    print(img_path)
    Img_data = cv2.imread(img_path)

    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    Gt_data = mat["image_info"][0][0][0][0][0]

    fidt_map1 = rdt_generate1(Img_data, Gt_data, 1)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map'), 'w') as hf:
        hf['fidt_map'] = fidt_map1
        hf['kpoint'] = kpoint

    fidt_map1 = fidt_map1
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)

    '''for visualization'''
    cv2.imwrite(img_path.replace('images', 'gt_show').replace('jpg', 'jpg'), fidt_map1)
