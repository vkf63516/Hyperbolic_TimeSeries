seq_len=720
train_epochs=30
patience=5
enc_in=21
manifold_type="Poincare"
num_basis=6
window_size=6
data_path=weather.csv

python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.2 \
  --hierarchy_weight 0.1 \
  --model_id Weather_$seq_len'_'$manifold_type'_'96_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis $num_basis \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 96 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --mstl_period 24 \
  --use_wandb \
  --learning_rate 1e-2 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --fine_period 144 \
  --coarse_period 1008 \
  --window_size 4 \
  --use_multi_horizon


python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.1 \
  --hierarchy_weight 0.01 \
  --model_id Weather_$seq_len'_'$manifold'_'192_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis $num_basis \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 192 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --mstl_period 24 \
  --use_wandb \
  --learning_rate 1e-2 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --fine_period 144 \
  --coarse_period 1008 \
  --window_size 4 \
  --use_multi_horizon

python run.py \
  --is_training 1 \
  --hierarchy_weight 0.05 \
  --model_id Weather_$seq_len'_'$manifold_type'_'336_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis $num_basis \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 336 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --mstl_period 24 \
  --fine_period 144 \
  --coarse_period 1008 \
  --use_wandb \
  --learning_rate 1e-2 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --window_size $window_size \
  --use_multi_horizon

python run.py \
  --is_training 1 \
  --hierarchy_weight 0.1 \
  --model_id Weather_$seq_len'_'$manifold_type'_'720_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis $num_basis \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 720 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --mstl_period 24 \
  --use_wandb \
  --learning_rate 1e-2 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --fine_period 144 \
  --coarse_period 1008 \
  --window_size $window_size \
  --use_multi_horizon
