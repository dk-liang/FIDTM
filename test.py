from __future__ import division
import warnings

from Networks.HR_Net.seg_hrnet import get_seg_model

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from image import *
from utils import *

import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args

warnings.filterwarnings('ignore')

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')


def main(args):
    if args['test_dataset'] == 'ShanghaiA':
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['test_dataset'] == 'ShanghaiB':
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['test_dataset'] == 'UCF_QNRF':
        test_file = './npydata/qnrf_test.npy'
    elif args['test_dataset'] == 'JHU':
        test_file = './npydata/jhu_val.npy'
    elif args['test_dataset'] == 'NWPU':
        test_file = './npydata/nwpu_val.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ])

    print(args['pre'])

    if not os.path.exists(args['task_id']):
        os.makedirs(args['task_id'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])
    print(args['best_pred'], args['start_epoch'])

    if args['preload_data'] == True:
        test_data = pre_data(val_list, args, train=False)
    else:
        test_data = val_list

    '''inference '''
    prec1, visi = validate(test_data, model, args)

    is_best = prec1 < args['best_pred']
    args['best_pred'] = min(prec1, args['best_pred'])

    print('\nThe visualizations are provided in ', args['task_id'])
    save_checkpoint({
        'arch': args['pre'],
        'state_dict': model.state_dict(),
        'best_prec1': args['best_pred'],
        'optimizer': optimizer.state_dict(),
    }, visi, is_best, args['task_id'])


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys


def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['task_id'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    if not os.path.exists('./local_eval/loc_file'):
        os.makedirs('./local_eval/loc_file')

    '''output coordinates'''
    f_loc = open("./local_eval/A_localization.txt", "w+")
    f_count = open("./local_eval/A_counting.txt", "w+")

    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):

        count = 0
        img = img.cuda()

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            d6 = model(img)[-1]
            '''only return counting'''
            # count = LMDS_with_counting(d6)
        '''return counting and coordinates'''
        count, f, f_count = LMDS_with_coordinates(d6, i + 1, f_loc, f_count)

        gt_count = torch.sum(kpoint).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 1 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
            visi.append(
                [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
                 fname])
            index += 1

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae, visi


def resever_fidt_map(input_img):
    pre_0 = input_img[0]
    pre_1 = input_img[1]
    pre_2 = input_img[2]
    pre_3 = input_img[3]

    pre_up = torch.cat([pre_0, pre_1], 2)
    pre_down = torch.cat([pre_2, pre_3], 2)
    input_img = torch.cat([pre_up, pre_down], 1).unsqueeze(0)
    return input_img



def LMDS_with_coordinates(input, fname, f_loc, f_count):
    input_max = torch.max(input).item()

    rate = 1
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
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])

        cv2.circle(point_map, (w, h), 2, (0, 255, 0))

    cv2.imwrite('1.jpg', point_map)
    f_loc.write('{} {} '.format(fname, count))
    # f_loc.write('{} {} '.format(fname.split('.')[0].split('_')[1], count))

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    count_value = (fname, count)
    s_count = str(count_value).strip('(').strip(')').replace(',', ' ').strip('\'').replace('\'  ', ' ').replace(']])',
                                                                                                                '')
    f_count.write(s_count)
    f_count.write('\n')

    return count, f_loc, f_count


'''only return counting'''

def LMDS_with_counting(input, threshold=100):
    input[input < 0] = 0
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    input[input < threshold / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1

    count = torch.sum(input).item()

    return count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
