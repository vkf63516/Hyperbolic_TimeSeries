#!/bin/bash
#SBATCH --job-name=ETTH1-P
#SBATCH --output=%x.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --time=02:00:00

echo "Job started at $(date)"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi

# ============================================================
# STEP 1: AGGRESSIVE CLEANUP (only YOUR processes)
# ============================================================
echo "========================================================================"
echo "CLEANING UP GPU"
echo "========================================================================"

# Kill only YOUR Python processes (not others')
#pkill -9 -u $USER python 2>/dev/null || true
#sleep 3

export PROJECT_ROOT=/project/mx6/sp3463

# Load modules
module purge > /dev/null 2>&1
module load wulver
module load bright
module load GCC/13.3.0
module load CUDA/12.6.0
module load Miniforge3


# Activate the environment
conda activate $PROJECT_ROOT/conda_envs/himenv

python /mmfs1/project/mx6/sp3463/HTS-S/run.py \
  --is_training 1 \
  --model_id ETTh1_Poincare_Original_96 \
  --model HyperbolicForecasting \
  --data ETTh1_decomposition \
  --root_path /mmfs1/project/mx6/sp3463/HTS-v/dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --encode_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --patience 5 \
  --use_decomposition \
  --enc_in 7 \
  --num_basis 10 \
  --manifold_type Poincare \
  --curvature 1.0 \
  --mstl_period 24
  
python /mmfs1/project/mx6/sp3463/HTS-S/run.py \
  --is_training 1 \
  --model_id ETTh1_Poincare_MovingWindow_96 \
  --model HyperbolicForecasting \
  --data ETTh1_decomposition \
  --root_path /mmfs1/project/mx6/sp3463/HTS-v/dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --encode_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --patience 5 \
  --use_decomposition \
  --enc_in 7 \
  --num_basis 10 \
  --manifold_type Poincare \
  --curvature 1.0 \
  --mstl_period 24 \
  --use_moving_window \
  --window_size 5