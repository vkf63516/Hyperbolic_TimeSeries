seq_len=720
train_epochs=30
patience=6
enc_in=137
manifold_type="Poincare"
data_path=Solar.csv
window_size=4
num_basis=6

python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.2 \
  --hierarchy_weight 0.001 \
  --model_id Solar_$seq_len'_'$manifold_type'_'96_exp1_Segment \
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
  --window_size 5 \
  --use_moving_window


python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.1 \
  --hierarchy_weight 0.1 \
  --model_id Solar_$seq_len'_'$manifold'_'192_exp1_Segment \
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
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --fine_period 144 \
  --coarse_period 1008 \
  --window_size 6 \
  --use_moving_window


python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.1 \
  --hierarchy_weight 0.001 \
  --model_id Solar_$seq_len'_'$manifold_type'_'336_exp1_Segment \
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
  --window_size 5 \
  --use_moving_window


python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.1 \
  --hierarchy_weight 0.0001 \
  --model_id Solar_$seq_len'_'$manifold_type'_'720_exp1_Segment \
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
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --fine_period 144 \
  --coarse_period 1008 \
  --window_size 6 \
  --use_moving_window
