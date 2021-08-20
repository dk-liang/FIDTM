import os
import math
import numpy as np
import time
import random
import shutil
import cv2
from PIL import Image

import sys
sys.setrecursionlimit(20000)  #

# Hungarian method for bipartite graph
def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]

    def dfs(u):
        for v in graph[u]:
            if vis[v]:
                continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum):
            vis[i] = False
        if dfs(a):
            ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign


def read_pred_and_gt(pred_file, gt_file):
    # read pred
    pred_data = {}
    with open(pred_file) as f:

        id_read = []
        for line in f.readlines():
            #print(line)
            line = line.strip().split(' ')

            # check1

            if len(line) < 2 or len(line) % 2 != 0 or (len(line) - 2) / 2 != int(line[1]):
                flagError = True
                sys.exit(1)

            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            id_read.append(idx)

            points = []
            if num > 0:
                points = np.array(line_data[2:]).reshape(((len(line) - 2) // 2, 2))
                pred_data[idx] = {'num': num, 'points': points}
            else:
                pred_data[idx] = {'num': num, 'points': []}

    # read gt
    gt_data = {}
    with open(gt_file) as f:
        for line in f.readlines():
            line = line.strip().split(' ')

            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            points_r = []
            if num > 0:
                points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                gt_data[idx] = {'num': num, 'points': points_r[:, 0:2], 'sigma': points_r[:, 2:4],
                                'level': points_r[:, 4]}
            else:
                gt_data[idx] = {'num': 0, 'points': [], 'sigma': [], 'level': []}

    return pred_data, gt_data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
