import argparse
import os
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import gc 

parser = argparse.ArgumentParser(description="Hyperbolic TimeSeries with orthogonalMSTL")
# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='HyperbolicForecasting', help='model name')
# orthogonalMSTL arguments
parser.add_argument('--gradient_truncation_K', type=int, default=6, 
                    help='truncation steps for BPTT in hyperbolic forecasting')
# NEW: Segment-level vs Point-level hyperbolic embeddings
parser.add_argument('--share_feature_weights', action='store_true', default=False,
                    help='share weights across features (for high-D data)')
parser.add_argument('--mstl_period', type=int, default=24,
                    help='MSTL period for segmentation (None=auto-detect from data frequency)')
parser.add_argument('--use_moving_window', action='store_true', default=False,
                    help='use moving window approach (True) or regular approach (False)')

parser.add_argument('--use_segments', action='store_true', default=False,
                    help='use segment-level hyperbolic embeddings (True) or point-level (False)')
parser.add_argument('--use_decomposition', action='store_true', default=False,
                    help='Use orthogonalMSTL decomposition')
parser.add_argument('--num_basis', type=int, default=10,
                    help='number of basis components for orthogonalMSTL')
parser.add_argument('--use_wandb', action='store_true', 
                    help='use wandb for experiment tracking')
parser.add_argument("--use_attention_pooling", action="store_true", default=False,
                    help="uses attention to give more relevance to specific timesteps")
parser.add_argument('--orthogonal_lr', type=float, default=1e-3,
                    help='learning rate for orthogonal basis fitting')
parser.add_argument('--orthogonal_iters', type=int, default=300,
                    help='number of optimization iterations for basis')

# Data processing

parser.add_argument('--log_interval', type=int, default=100,
                    help='logging interval during training')
parser.add_argument('--save_freq', type=int, default=10,
                    help='checkpoint save frequency (epochs)')

# Hyperbolic Space

parser.add_argument('--embed_dim', type=int, default=32, help='hyperbolic embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=128, help='mamba hidden dimension')
parser.add_argument('--curvature', type=float, default=1.0, help='negative number for hyperbolic curvature')
# Data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument("--manifold_type", type=str, default="Lorentzian", help="use either Lorentzian, Poincare, or Euclidean")
parser.add_argument('--features', type=str, default='M', help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate")
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--embed', type=str, default='fixed',help='time features encoding, options:[timeF, fixed, learned]')
# optimization
parser.add_argument('--use_revin', action='store_true', default=False, help='RevIN')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiment runs')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:fine, b:business days, w:coarse, m:monthly]')

# Replace the line with:
parser.add_argument('--use_tensorboard', action='store_true', default=False, 
                    help='use TensorBoard for logging')
# GPU
parser.add_argument('--use_amp', action='store_true', help='use AMP', default=False)
parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1', help='multi-gpu device ids')

args = parser.parse_args()

fix_seed_list = range(2023, 2033)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
mode_str = "Segment-level" if args.use_segments else "Point-level"
print(f'\n{"="*60}')
print(f'Hyperbolic Embedding Mode: {mode_str}')
print(f'{"="*60}\n')

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        random.seed(fix_seed_list[ii])
        torch.manual_seed(fix_seed_list[ii])
        np.random.seed(fix_seed_list[ii])
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_{}_eb{}_{}_{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.data_path,
            args.features,
            args.seq_len,
            args.pred_len,
            args.des,
            args.embed,
            args.manifold_type,
            ii,
            fix_seed_list[ii])

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        # if args.do_predict:
        #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_{}_eb{}_{}_{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.data_path,
        args.features,
        args.seq_len,
        args.pred_len,
        args.des,
        args.embed,
        args.manifold_type,
        ii,
        fix_seed_list[ii])
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
