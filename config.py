import argparse

parser = argparse.ArgumentParser(description='FIDTM')


parser.add_argument('--dataset', type=str, default='ShanghaiA',
                    help='choice train dataset')


parser.add_argument('--save_path', type=str, default='save_file/A_baseline',
                    help='save checkpoint directory')
parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=200,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')
parser.add_argument('--epochs', type=int, default=3000,
                    help='end epoch for training')
parser.add_argument('--pre', type=str, default=None,
                    help='pre-trained model directory')
# parser.add_argument('--pre', type=str, default='./model_best_qnrf.pth',
#                     help='pre-trained model directory')


parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--crop_size', type=int, default=256,
                    help='crop size for training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='1',
                    help='gpu id')
parser.add_argument('--lr', type=float, default= 1e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')
parser.add_argument('--preload_data', type=bool, default=True,
                    help='preload data. ')
parser.add_argument('--visual', type=bool, default=False,
                    help='visual for bounding box. ')

'''video demo'''
parser.add_argument('--video_path', type=str, default=None,
                    help='input video path ')

args = parser.parse_args()
return_args = parser.parse_args()
