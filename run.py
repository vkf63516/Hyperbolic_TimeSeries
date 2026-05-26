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

def setup_device():
    if torch.cuda.is_available():
        try:
            torch.cuda. empty_cache()
            device = torch.device('cuda')
            print("Using GPU")
        except: 
            device = torch.device('cpu')
            print("GPU busy, using CPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

parser = argparse.ArgumentParser(description="Hyperbolic TimeSeries with orthogonalMSTL")
# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='HyperbolicForecasting', help='model name')
# orthogonalMSTL arguments
parser.add_argument('--gradient_truncation_K', type=int, default=6, 
                    help='truncation steps for BPTT in hyperbolic forecasting')
# NEW: Segment-level vs Point-level hyperbolic encodedings
parser.add_argument('--hyperbolic_weight', type=float, default=0.2, help='temporal consistency weight')
parser.add_argument('--hierarchy_weight', type=float, default=0.0001, help='hierarchy weight')
parser.add_argument('--use_multi_horizon', action='store_true', default=False, 
                    help='one shot forecaster for Hyperbolic')
parser.add_argument('--share_feature_weights', action='store_true', default=False,
                    help='share weights across features (for high-D data)')
parser.add_argument('--mstl_period', type=int, default=24,
                    help='MSTL period for segmentation (None=auto-detect from data frequency)')
parser.add_argument('--use_moving_window', action='store_true', default=False,
                    help='use moving window approach (True) or regular approach (False)')

parser.add_argument('--use_segments', action='store_true', default=True,
                    help='use segment-level hyperbolic encodedings (True) or point-level (False)')
# ============================================
parser.add_argument('--use_learnable_decomposition', action='store_true', default=False,
                    help='use learnable Conv1D decomposition')
parser.add_argument('--use_no_decomposition', action='store_true', default=False,
                    help='no decomposition')

parser.add_argument('--fine_period', type=int, default=24,
                    help='fine-grained seasonal period (e.g., 24 for daily in hourly data)')
parser.add_argument('--coarse_period', type=int, default=168,
                    help='coarse-grained seasonal period (e.g., 168 for weekly in hourly data)')
parser.add_argument('--trend_period', type=int, default=336)
parser.add_argument('--num_basis', type=int, default=6,
                    help='number of basis components for orthogonalMSTL')
parser.add_argument('--use_wandb', action='store_true', 
                    help='use wandb for experiment tracking')
parser.add_argument("--use_attention_pooling", action="store_true", default=False,
                    help="uses attention to give more relevance to specific timesteps")
# Data processing

parser.add_argument('--log_interval', type=int, default=100,
                    help='logging interval during training')
parser.add_argument('--save_freq', type=int, default=10,
                    help='checkpoint save frequency (epochs)')

# Hyperbolic Space

parser.add_argument('--encode_dim', type=int, default=32, help='hyperbolic encodeding dimension')
parser.add_argument('--hidden_dim', type=int, default=256, help='mamba hidden dimension')
parser.add_argument('--curvature', type=float, default=1.0, help='negative number for hyperbolic curvature')
# Data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument("--manifold_type", type=str, default="Poincare", help="use either Lorentzian, Poincare, or Euclidean")
parser.add_argument('--features', type=str, default='M', help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate")
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument("--result_file", type=str, default='result.txt', help='where to save results in project')
# forecasting task
parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--encode', type=str, default='fixed',help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--window_size', type=int, default=None, help='size of window segments for average velocity')
# optimization
parser.add_argument('--inverse', action='store_true', default=False, help='inverse transform')
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
def set_seed(seed):
    """Fully deterministic seed setting."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
print(f'Hyperbolic encodeding Mode: {mode_str}')
print(f'{"="*60}\n')

Exp = Exp_Main
#if args.is_training:
#    all_mse = []
#    all_mae = []
#    all_settings = []
#
#    for ii in range(args.itr):
#        seed = fix_seed_list[ii]
#        set_seed(seed)
#
#        setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_{}_eb{}_{}_{}_seed{}'.format(
#            args.model_id,
#            args.model,
#            args.data,
#            args.data_path,
#            args.features,
#            args.seq_len,
#            args.pred_len,
#            args.des,
#            args.encode,
#            args.manifold_type,
#            ii,
#            seed)
#
#        all_settings.append(setting)
#        exp = Exp(args)
#        device = setup_device()
#
#        print(f'\n>>> Run {ii+1}/{args.itr} | Seed {seed} <<<')
#        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#        exp.train(setting)
#
#        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#        metrics = exp.test(setting)
#
#        if metrics is not None:
#            mse, mae = metrics
#            all_mse.append(mse)
#            all_mae.append(mae)
#            print(f'Run {ii+1} | Seed {seed}: MSE={mse:.4f}, MAE={mae:.4f}')
#        else:
#            print(f'WARNING: Run {ii+1} returned no metrics - check test() return value')
#
#        torch.cuda.empty_cache()
#        gc.collect()
#
#    # ============================================================
#    # Aggregate results across all runs
#    # ============================================================
#    if len(all_mse) > 1:
#        mse_arr = np.array(all_mse)
#        mae_arr = np.array(all_mae)
#
#        mse_mean = np.mean(mse_arr)
#        mse_std  = np.std(mse_arr, ddof=1)   # ddof=1 = sample std
#        mae_mean = np.mean(mae_arr)
#        mae_std  = np.std(mae_arr, ddof=1)
#
#        # 95% confidence interval
#        n = len(mse_arr)
#        mse_ci = stats.t.interval(
#            0.95, df=n-1,
#            loc=mse_mean,
#            scale=stats.sem(mse_arr)
#        )
#        mae_ci = stats.t.interval(
#            0.95, df=n-1,
#            loc=mae_mean,
#            scale=stats.sem(mae_arr)
#        )
#
#        print(f'\n{"="*60}')
#        print(f'FINAL RESULTS: {args.data} | pred_len={args.pred_len} | {n} runs')
#        print(f'{"="*60}')
#        print(f'Individual MSEs : {[f"{v:.4f}" for v in mse_arr]}')
#        print(f'Individual MAEs : {[f"{v:.4f}" for v in mae_arr]}')
#        print(f'MSE : {mse_mean:.4f} ± {mse_std:.4f}')
#        print(f'MAE : {mae_mean:.4f} ± {mae_std:.4f}')
#        print(f'MSE 95% CI : [{mse_ci[0]:.4f}, {mse_ci[1]:.4f}]')
#        print(f'MAE 95% CI : [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]')
#        print(f'{"="*60}\n')
#
#        # Save everything to file
#        result_path = (
#            f'./results_multiseed/'
#            f'{args.model_id}_{args.data}_pl{args.pred_len}_{n}runs.txt'
#        )
#        os.makedirs('./results_multiseed/', exist_ok=True)
#
#        with open(result_path, 'w') as f:
#            f.write(f'Dataset     : {args.data}\n')
#            f.write(f'Data path   : {args.data_path}\n')
#            f.write(f'pred_len    : {args.pred_len}\n')
#            f.write(f'Model       : {args.model}\n')
#            f.write(f'N runs      : {n}\n')
#            f.write(f'Seeds       : {fix_seed_list[:n]}\n')
#            f.write(f'\nIndividual results:\n')
#            for i, (s, m, ma) in enumerate(zip(fix_seed_list[:n], mse_arr, mae_arr)):
#                f.write(f'  Seed {s}: MSE={m:.4f}, MAE={ma:.4f}\n')
#            f.write(f'\nAggregated:\n')
#            f.write(f'  MSE : {mse_mean:.4f} ± {mse_std:.4f}\n')
#            f.write(f'  MAE : {mae_mean:.4f} ± {mae_std:.4f}\n')
#            f.write(f'  MSE 95% CI : [{mse_ci[0]:.4f}, {mse_ci[1]:.4f}]\n')
#            f.write(f'  MAE 95% CI : [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]\n')
#
#        print(f'Results saved to {result_path}')
#
#else:
#    # Inference only - single run
#    ii = 0
#    seed = fix_seed_list[ii]
#    set_seed(seed)
#
#    setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_{}_eb{}_{}_{}_seed{}'.format(
#        args.model_id,
#        args.model,
#        args.data,
#        args.data_path,
#        args.features,
#        args.seq_len,
#        args.pred_len,
#        args.des,
#        args.encode,
#        args.manifold_type,
#        ii,
#        seed)
#
#    device = setup_device()
#    exp = Exp(args)
#    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#    exp.test(setting, test=1)
#    torch.cuda.empty_cache()
if args.is_training:
    for ii in range(args.itr):
        set_seed(fix_seed_list[ii])
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
            args.encode,
            args.manifold_type,
            ii,
            fix_seed_list[ii])

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        device = setup_device()

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
        args.encode,
        args.manifold_type,
        ii,
        fix_seed_list[ii])
    device = setup_device()

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
