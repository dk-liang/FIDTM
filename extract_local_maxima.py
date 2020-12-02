
import torch
import cv2
import numpy as np
import torch.nn as nn


def draw_pred_point(input, fname, rate, coord_list, crop_size, refine):
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    input[input < 100.0 / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1
    count = int(torch.sum(input).item())

    pred_kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()
    pred_coor = np.nonzero(pred_kpoint)

    point_map = np.zeros((int(input.shape[2] * rate), int(input.shape[3] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])

    if refine == False:
        for i in range(0, len(pred_coor[0])):
            h = int(pred_coor[0][i] * rate)
            w = int(pred_coor[1][i] * rate)
            coord_list.append([w, h])

            cv2.circle(point_map, (w, h), 2, (0, 255, 0))

    elif refine == True:
        for i in range(0, len(pred_coor[0])):
            h = int((pred_coor[0][i] / 2 + crop_size[1]) * rate)
            w = int((pred_coor[1][i] / 2 + crop_size[0]) * rate)
            coord_list.append([w, h])

            cv2.circle(point_map, (w, h), 2, (0, 255, 0))

    return count, coord_list