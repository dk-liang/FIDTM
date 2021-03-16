import argparse



parser = argparse.ArgumentParser(description='AutoScale for regression based-method')


# Data specifications
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

# Model specifications
parser.add_argument('--test_dataset', type=str, default='UCF_QNRF',
                    help='choice train dataset')
# parser.add_argument('--pre', type=str, default=None,
#                     help='pre-trained model directory')
parser.add_argument('--pre', type=str, default='./model/model_best_6.9.pth',
                    help='pre-trained model directory')

# Optimization specifications
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--epochs', type=int, default=20000,
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='1',
                    help='gpu id')
parser.add_argument('--lr_step', type=int, default=4500,
                         help='drop learning rate by 10.')
parser.add_argument('--preload', type=bool, default=True,
                         help='load predata')

#SSIM config
#SSIM config
parser.add_argument('--ssim_loss', type=str, default="True",
                    help='ssim type')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='alpha value')
parser.add_argument('--gama', type=float, default=1.0,
                    help='gama value')
parser.add_argument('--mse_size_average', type=str, default="False",
                    help='mse_size_average')
parser.add_argument('--IA_ssim', type=str, default="True",
                    help='IA_ssim')
parser.add_argument('--focul_loss', type=str, default="False",
                    help='focul_loss')
parser.add_argument('--mix_reg_loc', type=str, default=False,
                    help='mix_reg_loc')

# nni config
parser.add_argument('--lr', type=float, default= 1e-4,
                    help='learning rate')
parser.add_argument('--center_lr', type=float, default=1e-4,
                    help='learning rate of rate model')
parser.add_argument('--random_noise', type=float, default=1,
                         help='random_noise')
parser.add_argument('--rdt', type=int, default=1,
                         help='rdt type')

args = parser.parse_args()
return_args =  parser.parse_args()