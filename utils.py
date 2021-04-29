import h5py
import torch
import shutil
import numpy as np
import cv2
import os
import random


def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    density_map[density_map < 0] = 0

    gt_data = 255 * gt_data / np.max(gt_data)
    gt_data = gt_data[0][0]
    gt_data = gt_data.astype(np.uint8)
    gt_data = cv2.applyColorMap(gt_data, 2)

    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)

    result_img = np.hstack((gt_data, density_map))

    cv2.imwrite(os.path.join('.', output_dir, fname).replace('.jpg', '.jpg'), result_img)


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, visi, is_best, save_path, filename='checkpoint.pth'):
    torch.save(state, './' + str(save_path) + '/' + filename)
    if is_best:
        shutil.copyfile('./' + str(save_path) + '/' + filename, './' + str(save_path) + '/' + 'model_best.pth')

    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, str(save_path), fname[0])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 输入固定情况下用true
