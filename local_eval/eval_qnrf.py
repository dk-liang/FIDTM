import os
import sys
import numpy as np
from scipy import spatial as ss
import pdb

import cv2
from utils import hungarian, read_pred_and_gt, AverageMeter, AverageCategoryMeter
import argparse
import scipy.io
import math

flagError = False
# id_std = [i for i in range(3110,3610,1)]
# id_std[59] = 3098
id_std = [i for i in range(1, 335, 1)]

num_classes = 1


def compute_metrics(dist_matrix, match_matrix, pred_num, gt_num, sigma, level):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]
    level_list = level[tp_gt_index]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    tp_c = np.zeros([num_classes])
    fn_c = np.zeros([num_classes])

    for i_class in range(num_classes):
        tp_c[i_class] = (level[tp_gt_index] == i_class).sum()
        fn_c[i_class] = (level[fn_gt_index] == i_class).sum()

    return tp, fp, fn, tp_c, fn_c


def main(gt_file, pred_file, index):
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter(), }
    metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(),
                 'tp_c': AverageCategoryMeter(num_classes), 'fn_c': AverageCategoryMeter(num_classes)}
    metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(),
                 'tp_c': AverageCategoryMeter(num_classes), 'fn_c': AverageCategoryMeter(num_classes)}

    pred_data, gt_data = read_pred_and_gt(pred_file, gt_file)
    for i_sample in id_std:

        # init               
        gt_p, pred_p, fn_gt_index, tp_pred_index, fp_pred_index = [], [], [], [], []
        tp_s, fp_s, fn_s, tp_l, fp_l, fn_l = [0, 0, 0, 0, 0, 0]
        tp_c_l = np.zeros([num_classes])
        fn_c_l = np.zeros([num_classes])

        if gt_data[i_sample]['num'] == 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
            fp_pred_index = np.array(range(pred_p.shape[0]))
            fp_s = fp_pred_index.shape[0]
            fp_l = fp_pred_index.shape[0]

        if pred_data[i_sample]['num'] == 0 and gt_data[i_sample]['num'] != 0:
            gt_p = gt_data[i_sample]['points']
            level = gt_data[i_sample]['level']
            fn_gt_index = np.array(range(gt_p.shape[0]))
            fn_s = fn_gt_index.shape[0]
            fn_l = fn_gt_index.shape[0]
            for i_class in range(num_classes):
                fn_c_l[i_class] = (level[fn_gt_index] == i_class).sum()

        if gt_data[i_sample]['num'] != 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
            gt_p = gt_data[i_sample]['points']
            sigma_l = gt_data[i_sample]['sigma'][:, 1]
            level = gt_data[i_sample]['level']

            # dist
            dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)

            # sigma_s and sigma_l
            tp_l, fp_l, fn_l, tp_c_l, fn_c_l = compute_metrics(dist_matrix, match_matrix, pred_p.shape[0],
                                                               gt_p.shape[0], sigma_l, level)

        metrics_l['tp'].update(tp_l)
        metrics_l['fp'].update(fp_l)
        metrics_l['fn'].update(fn_l)
        metrics_l['tp_c'].update(tp_c_l)
        metrics_l['fn_c'].update(fn_c_l)

        gt_count, pred_cnt = gt_data[i_sample]['num'], pred_data[i_sample]['num']
        s_mae = abs(gt_count - pred_cnt)
        s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)
        cnt_errors['mae'].update(s_mae)
        cnt_errors['mse'].update(s_mse)

        if gt_count != 0:
            s_nae = abs(gt_count - pred_cnt) / gt_count
            cnt_errors['nae'].update(s_nae)

    ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
    ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
    f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l)
    ar_c_l = metrics_l['tp_c'].sum / (metrics_l['tp_c'].sum + metrics_l['fn_c'].sum + 1e-20)

    print('-----Localization performance-----  ', index)
    print('AP_large: ' + str(ap_l))
    print('AR_large: ' + str(ar_l))
    print('F1m_large: ' + str(f1m_l))

    return ap_l, ar_l, f1m_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIDTM')
    parser.add_argument('--data_path', default= '/home/dkliang/projects/synchronous/dataset/UCF-QNRF_ECCV18', type=str, help='the QNRF data path')
    args = parser.parse_args()


    img_test_path = args.data_path + '/Test/'
    gt_test_path = args.data_path + '/Test/'
    img_test = []
    gt_test = []

    for file_name in os.listdir(img_test_path):
        if file_name.split('.')[1] == 'jpg':
            img_test.append(img_test_path + file_name)

    for file_name in os.listdir(gt_test_path):
        if file_name.split('.')[1] == 'mat':
            gt_test.append(file_name)

    img_test.sort()
    gt_test.sort()

    ap_l_list = []
    ar_l_list = []
    f1m_l_list = []
    for i in range(1, 101):
        print("start process sigma = ", i)
        f = open('./point_files/qnrf_gt.txt', 'w+')
        k = 0
        for img_path in img_test:
            Gt_data = scipy.io.loadmat(gt_test_path + gt_test[k])
            Gt_data = Gt_data['annPoints']

            f.write('{} {} '.format(k + 1, len(Gt_data)))
            ''' fixed sigma'''
            sigma_s = 4
            sigma_l = i
            for data in Gt_data:
                f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1))
            f.write('\n')

            k = k + 1
        f.close()
        gt_file = './point_files/qnrf_gt.txt'
        pred_file = './point_files/qnrf_pred_fidt.txt'
        ap_l, ar_l, f1m_l = main(gt_file, pred_file, i)

        ap_l_list.append(ap_l)
        ar_l_list.append(ar_l)
        f1m_l_list.append(f1m_l)

    print(np.mean(ap_l_list), np.mean(ar_l_list), np.mean(f1m_l_list))
