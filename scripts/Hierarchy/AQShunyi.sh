#!/bin/bash
#SBATCH --job-name=AQShunyi-M-P
#SBATCH --output=%x.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --time=00:10:00
#SBATCH --exclude=n0046,n0025,n0064,n0066


# ============================================================
# STEP 1: AGGRESSIVE CLEANUP (only YOUR processes)
# ============================================================
echo "========================================================================"
echo "CLEANING UP GPU"
echo "========================================================================"

# Kill only YOUR Python processes (not others')
#pkill -9 -u $USER python 2>/dev/null || true
#sleep 3


# Load modules


# Activate the environment

seq_len=720
train_epochs=30
patience=5
enc_in=11
manifold_type="Poincare"
data_path=AQShunyi.csv
num_basis=10
encode_dim=64
window_size=2

#python -u /mmfs1/project/mx6/sp3463/vis-HSF/run.py \
#  --is_training 1 \
#  --model_id AQShunyi_${seq_len}_${manifold_type}_96_exp1_Segment \
#  --model HyperbolicForecasting \
#  --data custom \
#  --root_path /mmfs1/project/mx6/sp3463/HTS-v/dataset/ \
#  --data_path $data_path \
#  --features M \
#  --num_basis $num_basis \
#  --label_len 0 \
#  --seq_len $seq_len \
#  --pred_len 96 \
#  --lradj "type3" \
#  --encode_dim $encode_dim \
#  --hidden_dim 256 \
#  --batch_size 32 \
#  --use_wandb \
#  --learning_rate 1e-3 \
#  --train_epochs $train_epochs \
#  --use_learnable_decomposition \
#  --enc_in $enc_in \
#  --patience $patience \
#  --manifold_type $manifold_type \
#  --use_revin \
#  --use_segments \
#  --window_size $window_size \
#  --use_moving_window


python  visualize_poincare_embeddings.py \
    --embeddings_path ./checkpoints/AQShunyi_720_Poincare_96_exp1_Segment_HyperbolicForecasting_custom_AQShunyi.csv_ftM_sl720_pl96_exp_ebfixed_Poincare_0_seed2023/best_epoch_embeddings.pt \
    --num_subgraphs 5 \
    --max_nodes 8 \
    --save_dir ./AQShunyi_viz
    
#python /mmfs1/project/mx6/sp3463/HSF/run.py \
#  --is_training 1 \
#  --model_id AQShunyi_$seq_len'_'$manifold_type'_'192_exp1_Segment \
#  --model HyperbolicForecasting \
#  --data custom \
#  --root_path /mmfs1/project/mx6/sp3463/HTS-v/dataset/ \
#  --data_path $data_path \
#  --features M \
#  --num_basis $num_basis \
#  --label_len 0 \
#  --seq_len $seq_len \
#  --pred_len 192 \
#  --lradj "type3" \
#  --encode_dim $encode_dim \
#  --hidden_dim 256 \
#  --batch_size 32 \
#  --use_wandb \
#  --learning_rate 1e-3 \
#  --train_epochs $train_epochs \
#  --use_learnable_decomposition \
#  --enc_in $enc_in \
#  --patience $patience \
#  --manifold_type $manifold_type \
#  --use_revin \
#  --use_segments \
#  --window_size $window_size \
#  --use_moving_window


#python /mmfs1/project/mx6/sp3463/HSF/run.py \
#  --is_training 1 \
#  --model_id AQShunyi_$seq_len'_'$manifold_type'_'336_exp1_Segment \
#  --model HyperbolicForecasting \
#  --data custom \
#  --root_path /mmfs1/project/mx6/sp3463/HTS-v/dataset/ \
#  --data_path $data_path \
#  --features M \
#  --num_basis $num_basis \
#  --label_len 0 \
#  --seq_len $seq_len \
#  --pred_len 336 \
#  --lradj "type3" \
#  --encode_dim $encode_dim \
#  --hidden_dim 256 \
#  --batch_size 32 \
#  --use_wandb \
#  --learning_rate 1e-3 \
#  --train_epochs $train_epochs \
#  --use_learnable_decomposition \
# --enc_in $enc_in \
# --patience $patience \
#  --manifold_type $manifold_type \
#  --use_revin \
#  --use_segments \
#  --window_size $window_size \
#  --use_moving_window

#python /mmfs1/project/mx6/sp3463/HSF/run.py \
#  --is_training 1 \
#  --model_id AQShunyi_$seq_len'_'$manifold_type'_'720_exp1_Segment \
#  --model HyperbolicForecasting \
#  --data custom \
#  --root_path /mmfs1/project/mx6/sp3463/HTS-v/dataset/ \
#  --data_path $data_path \
#  --features M \
#  --num_basis $num_basis \
#  --label_len 0 \
#  --seq_len $seq_len \
#  --pred_len 720 \
#  --lradj "type3" \
#  --encode_dim $encode_dim \
#  --hidden_dim 256 \
#  --batch_size 32 \
#  --use_wandb \
#  --learning_rate 1e-3 \
#  --train_epochs $train_epochs \
#  --use_learnable_decomposition \
#  --enc_in $enc_in \
#  --patience $patience \
#  --manifold_type $manifold_type \
#  --use_revin \
#  --use_segments \
#  --window_size $window_size \
#  --use_moving_window
