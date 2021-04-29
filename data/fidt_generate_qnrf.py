import os
import time

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import math
import torch

'''please set the dataset path'''
root = '/home/dkliang/projects/synchronous/dataset/UCF-QNRF_ECCV18'

img_train_path = root + '/Train/'
gt_train_path = root + '/Train/'
img_test_path = root + '/Test/'
gt_test_path = root + '/Test/'

save_train_img_path = root + '/train_data/images/'
save_train_gt_path = root + '/train_data/gt_fidt_map/'
save_test_img_path = root + '/test_data/images/'
save_test_gt_path = root + '/test_data/gt_fidt_map/'

if not os.path.exists(save_train_img_path):
    os.makedirs(save_train_img_path)

if not os.path.exists(save_train_gt_path):
    os.makedirs(save_train_gt_path)

if not os.path.exists(save_train_img_path.replace('images', 'gt_show_fidt')):
    os.makedirs(save_train_img_path.replace('images', 'gt_show_fidt'))

if not os.path.exists(save_test_img_path):
    os.makedirs(save_test_img_path)

if not os.path.exists(save_test_gt_path):
    os.makedirs(save_test_gt_path)

if not os.path.exists(save_test_img_path.replace('images', 'gt_show_fidt')):
    os.makedirs(save_test_img_path.replace('images', 'gt_show_fidt'))

distance = 1
img_train = []
gt_train = []
img_test = []
gt_test = []

for file_name in os.listdir(img_train_path):
    if file_name.split('.')[1] == 'jpg':
        img_train.append(file_name)

for file_name in os.listdir(gt_train_path):
    if file_name.split('.')[1] == 'mat':
        gt_train.append(file_name)

for file_name in os.listdir(img_test_path):
    if file_name.split('.')[1] == 'jpg':
        img_test.append(file_name)

for file_name in os.listdir(gt_test_path):
    if file_name.split('.')[1] == 'mat':
        gt_test.append(file_name)

img_train.sort()
gt_train.sort()
img_test.sort()
gt_test.sort()
# print(img_train)
# print(gt_train)
print(len(img_train), len(gt_train), len(img_test), len(gt_test))


''''generate fidt map'''
def fidt_generate1(im_data, gt_data, lamda):
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


'''for training dataset'''
# for k in range(len(img_train)):
#
#     Img_data = cv2.imread(img_train_path + img_train[k])
#     Gt_data = scipy.io.loadmat(gt_train_path + gt_train[k])
#     rate = 1
#     rate_1 = 1
#     rate_2 = 1
#     flag = 0
#     if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
#         rate_1 = 2048.0 / Img_data.shape[1]
#         flag = 1
#     if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
#         rate_1 = 2048.0 / Img_data.shape[0]
#         flag = 1
#     Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_1)
#
#     min_shape = 512.0
#     if Img_data.shape[1] <= Img_data.shape[0] and Img_data.shape[1] <= min_shape:
#         rate_2 = min_shape / Img_data.shape[1]
#     elif Img_data.shape[0] <= Img_data.shape[1] and Img_data.shape[0] <= min_shape:
#         rate_2 = min_shape / Img_data.shape[0]
#     Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_2)
#
#     rate = rate_1 * rate_2
#
#     Gt_data = Gt_data['annPoints']
#     Gt_data = Gt_data * rate
#     fidt_map = fidt_generate1(Img_data, Gt_data, 1)
#
#     kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
#     for count in range(0, len(Gt_data)):
#         if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
#             kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
#
#     new_img_path = (save_train_img_path + img_train[k])
#
#     mat_path = new_img_path.split('.jpg')[0]
#     gt_show_path = new_img_path.replace('images', 'gt_show_fidt')
#     h5_path = save_train_gt_path + img_train[k].replace('.jpg', '.h5')
#
#     print(img_train[k], np.sum(kpoint))
#
#     kpoint = kpoint.astype(np.uint8)
#     with h5py.File(h5_path, 'w') as hf:
#         hf['kpoint'] = kpoint
#         hf['fidt_map'] = fidt_map
#
#     cv2.imwrite(new_img_path, Img_data)
#
#     fidt_map = fidt_map
#     fidt_map = fidt_map / np.max(fidt_map) * 255
#     fidt_map = fidt_map.astype(np.uint8)
#     fidt_map = cv2.applyColorMap(fidt_map, 2)
#
#     result = fidt_map
#
#     cv2.imwrite(gt_show_path, result)

'''for testing dataset'''
for k in range(len(img_test)):
    Img_data = cv2.imread(img_test_path + img_test[k])
    Gt_data = scipy.io.loadmat(gt_test_path + gt_test[k])

    rate_1 = 1
    rate_2 = 1

    if Img_data.shape[1] > Img_data.shape[0] and Img_data.shape[1] >= 2048:
        rate_1 = 2048.0 / Img_data.shape[1]
    if Img_data.shape[0] > Img_data.shape[1] and Img_data.shape[0] >= 2048:
        rate_1 = 2048.0 / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_1)

    min_shape = 1024.0
    if Img_data.shape[1] <= Img_data.shape[0] and Img_data.shape[1] <= min_shape:
        rate_2 = min_shape / Img_data.shape[1]
    elif Img_data.shape[0] <= Img_data.shape[1] and Img_data.shape[0] <= min_shape:
        rate_2 = min_shape / Img_data.shape[0]
    Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_2)

    rate = rate_1 * rate_2

    print(img_test[k], Img_data.shape)

    Gt_data = Gt_data['annPoints']
    Gt_data = Gt_data * rate

    fidt_map = fidt_generate1(Img_data, Gt_data, 1)
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] += 1

    new_img_path = (save_test_img_path + img_test[k])

    mat_path = new_img_path.split('.jpg')[0]
    gt_show_path = new_img_path.replace('images', 'gt_show_fidt')
    h5_path = save_test_gt_path + img_test[k].replace('.jpg', '.h5')


    kpoint = kpoint.astype(np.uint8)
    with h5py.File(h5_path, 'w') as hf:
        hf['kpoint'] = kpoint
        hf['fidt_map'] = fidt_map

    cv2.imwrite(new_img_path, Img_data)

    fidt_map = fidt_map
    fidt_map = fidt_map / np.max(fidt_map) * 255
    fidt_map = fidt_map.astype(np.uint8)
    fidt_map = cv2.applyColorMap(fidt_map, 2)

    # result = np.hstack((mask_map,fidt_map))
    result = fidt_map

    cv2.imwrite(gt_show_path, result)
