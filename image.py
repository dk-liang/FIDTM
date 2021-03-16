import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2


def load_data_rdt(img_path, args, train=True):

    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_fidt_map')
    img = Image.open(img_path).convert('RGB')


    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            mask_map = np.asarray(gt_file['mask_map'])

            if args['rdt'] == 1:
                rdt_map = np.asarray(gt_file['fidt_map1'])
            break  # Success!
        except OSError:
            print(img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    rdt_map = rdt_map.copy()
    k = k.copy()
    mask_map = mask_map.copy()

    return img, rdt_map, k, mask_map

