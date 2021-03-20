import argparse

parser = argparse.ArgumentParser(description='FIDTM')


parser.add_argument('--train_dataset', type=str, default='ShanghaiA',
                    help='choice train dataset')

parser.add_argument('--task_id', type=str, default='save_file/A_baseline',
                    help='save checkpoint directory')
parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=200,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')


parser.add_argument('--test_dataset', type=str, default='ShanghaiA',
                    help='choice train dataset')
# parser.add_argument('--pre', type=str, default=None,
#                     help='pre-trained model directory')
parser.add_argument('--pre', type=str, default='./model/model_best_6.9.pth',
                    help='pre-trained model directory')


parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='1',
                    help='gpu id')
parser.add_argument('--lr', type=str, default='1e-4',
                    help='learning rate')
parser.add_argument('--preload_data', type=bool, default=True,
                    help='preload data. ')

args = parser.parse_args()
return_args = parser.parse_args()
