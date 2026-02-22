seq_len=720
train_epochs=30
patience=5
enc_in=184
manifold_type="Poincare"
data_path=pm2_5.csv
window_size=2

python run.py \
 --is_training 1 \
 --hyperbolic_weight 0.01 \
 --hierarchy_weight 0.0 \
 --model_id PM25_$seq_len'_'$manifold_type'_'96_exp1_Segment \
 --model HyperbolicForecasting \
 --data custom \
 --root_path ./time-series-dataset/dataset/ \
 --data_path $data_path \
 --features M \
 --num_basis 10 \
 --label_len 0 \
 --seq_len $seq_len \
 --pred_len 96 \
 --lradj "type3" \
 --encode_dim 64 \
 --hidden_dim 256 \
 --batch_size 32 \
 --use_wandb \
 --learning_rate 1e-3 \
 --train_epochs $train_epochs \
 --use_learnable_decomposition \
 --enc_in $enc_in \
 --patience $patience \
 --manifold_type $manifold_type \
 --use_revin \
 --use_segments \
 --fine_period 8 \
 --coarse_period 56 \
 --window_size $window_size \
 --use_multi_horizon


python run.py \
 --is_training 1 \
 --hyperbolic_weight 0.0 \
 --hierarchy_weight 0.0 \
 --model_id PM25_$seq_len'_'$manifold'_'192_exp1_Segment \
 --model HyperbolicForecasting \
 --data custom \
 --root_path ./time-series-dataset/dataset/ \
 --data_path $data_path \
 --features M \
 --num_basis 10 \
 --label_len 0 \
 --seq_len $seq_len \
 --pred_len 192 \
 --lradj "type3" \
 --encode_dim 64 \
 --hidden_dim 256 \
 --batch_size 32 \
 --use_wandb \
 --learning_rate 1e-3 \
 --train_epochs $train_epochs \
 --use_learnable_decomposition \
 --enc_in $enc_in \
 --patience $patience \
 --manifold_type $manifold_type \
 --use_revin \
 --use_segments \
 --fine_period 8 \
 --coarse_period 56 \
 --window_size $window_size \
 --use_multi_horizon

python run.py \
 --is_training 1 \
 --hyperbolic_weight 0.01 \
 --hierarchy_weight 0.0005 \
 --model_id PM25_$seq_len'_'$manifold_type'_'336_exp1_Segment \
 --model HyperbolicForecasting \
 --data custom \
 --root_path ./time-series-dataset/dataset/ \
 --data_path $data_path \
 --features M \
 --num_basis 10 \
 --label_len 0 \
 --seq_len $seq_len \
 --pred_len 336 \
 --lradj "type3" \
 --encode_dim 64 \
 --hidden_dim 256 \
 --batch_size 32 \
 --use_wandb \
 --learning_rate 1e-3 \
 --train_epochs $train_epochs \
 --use_learnable_decomposition \
 --enc_in $enc_in \
 --patience $patience \
 --manifold_type $manifold_type \
 --use_revin \
 --use_segments \
 --fine_period 8 \
 --coarse_period 56 \
 --window_size $window_size \
 --use_multi_horizon

python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.0 \
  --hierarchy_weight 0.0 \
  --model_id PM25_$seq_len'_'$manifold_type'_'720_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 720 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-4 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --fine_period 8 \
  --coarse_period 56 \
  --window_size $window_size \
  --use_multi_horizon
