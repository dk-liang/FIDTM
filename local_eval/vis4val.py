
import os
import sys
import numpy as np
from scipy import spatial as ss
import pdb

import cv2
from utils import hungarian, read_pred_and_gt, AverageMeter, AverageCategoryMeter

exp_name = './A_result'
train_flag = False

if train_flag == False:

    gt_file = './loc_file/A_gt.txt'
    pred_file = './loc_file/A_train_hrnet.txt'
    val_file = '../npydata/ShanghaiA_test.npy'
    with open(val_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    id_std = [i for i in range(1, 183, 1)]
else:
    gt_file = './loc_file/A_gt_train.txt'
    pred_file = './loc_file/A_train_loc.txt'
    val_file = '../npydata/ShanghaiA_train.npy'
    with open(val_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    id_std = [i for i in range(1, 301, 1)]

flagError = False


if not os.path.exists(exp_name):
    os.mkdir(exp_name)




def main():
    pred_data, gt_data = read_pred_and_gt(pred_file, gt_file)

    for i_sample in id_std:


        gt_p, pred_p, fn_gt_index, tp_pred_index, fp_pred_index, ap, ar = [], [], [], [], [], [], []

        if gt_data[i_sample]['num'] == 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
            fp_pred_index = np.array(range(pred_p.shape[0]))
            ap = 0
            ar = 0

        if pred_data[i_sample]['num'] == 0 and gt_data[i_sample]['num'] != 0:
            gt_p = gt_data[i_sample]['points']
            fn_gt_index = np.array(range(gt_p.shape[0]))
            sigma_l = gt_data[i_sample]['sigma'][:, 1]
            ap = 0
            ar = 0

        if gt_data[i_sample]['num'] != 0 and pred_data[i_sample]['num'] != 0:
            pred_p = pred_data[i_sample]['points']
            gt_p = gt_data[i_sample]['points']
            sigma_l = gt_data[i_sample]['sigma'][:, 1]
            level = gt_data[i_sample]['level']

            # dist
            dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
            for i_pred_p in range(pred_p.shape[0]):
                pred_dist = dist_matrix[i_pred_p, :]
                match_matrix[i_pred_p, :] = pred_dist <= sigma_l

            # hungarian outputs a match result, which may be not optimal.
            # Nevertheless, the number of tp, fp, tn, fn are same under different match results
            # If you need the optimal result for visualzation,
            # you may treat it as maximum flow problem.
            tp, assign = hungarian(match_matrix)
            fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
            tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
            tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
            fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

            cor_index = np.nonzero(assign)

            pre = tp_pred_index.shape[0] / (tp_pred_index.shape[0] + fp_pred_index.shape[0] + 1e-20)
            rec = tp_pred_index.shape[0] / (tp_pred_index.shape[0] + fn_gt_index.shape[0] + 1e-20)

        print(i_sample, val_list[i_sample-1], gt_p.shape[0], pred_p.shape[0])

        img = cv2.imread(val_list[i_sample-1])  # bgr

        print(len(cor_index[0]), len(cor_index[1]), max(cor_index[0]), max(cor_index[1]))

        for l in range(gt_p.shape[0]):
            gt_cor = gt_p[l]
            cv2.circle(img, (gt_cor[0], gt_cor[1]), 2, (0, 0, 255), 1)  # fn: red

        for l in range(len(cor_index[0])):
            pre_cor = pred_p[cor_index[0][l]]
            gt_cor = gt_p[cor_index[1][l]]

            gt_p[cor_index[1][l]] = pred_p[cor_index[0][l]]

            cv2.circle(img, (gt_cor[0], gt_cor[1]), 2, (0, 0, 255), 1)  # fn: red
            cv2.circle(img, (pre_cor[0], pre_cor[1]), 1, (0, 255, 0), 1)  # fp: blue

            cv2.line(img, (pre_cor[0], pre_cor[1]), (gt_cor[0], gt_cor[1]), (255, 0, 0), 1)

        # for l in range(gt_p.shape[0]):
        #     gt_cor = gt_p[l]
        #     cv2.circle(img, (gt_cor[0], gt_cor[1]), 2, (0, 255, 0), 1)  # fn: red


        point_r_value = 1
        thickness = 1
        # if gt_data[i_sample]['num'] != 0:
        #     for i in range(gt_p.shape[0]):
        #         cv2.circle(img, (gt_p[i][0], gt_p[i][1]), 2, (0, 0, 255), 1)  # fn: red
        #
        #
        # if pred_data[i_sample]['num'] != 0:
        #     for i in range(pred_p.shape[0]):
        #         if i in tp_pred_index:
        #             cv2.circle(img, (pred_p[i][0], pred_p[i][1]), point_r_value, (0, 255, 0), -1)  # tp: green
        #         else:
        #             cv2.circle(img, (pred_p[i][0], pred_p[i][1]), point_r_value , (255, 0, 0), -1)  # fp: blue

        # cv2.imwrite(exp_name + '/' + str(i_sample) + '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6] + '.bmp', img)
        cv2.imwrite(exp_name + '/' + str(i_sample) + '.bmp', img)

        if i_sample >3:
            break
        # break
if __name__ == '__main__':
    main()
