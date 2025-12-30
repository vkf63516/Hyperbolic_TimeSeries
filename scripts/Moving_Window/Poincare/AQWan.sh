seq_len=720
train_epochs=30
patience=5
enc_in=11
manifold_type="Poincare"
data_path=AQWan.csv
num_basis=10
window_size=2

python run.py \
  --is_training 1 \
  --model_id AQWan_$seq_len'_'$manifold_type'_'96_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom_decomposition \
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
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --window_size $window_size \
  --use_moving_window


python run.py \
  --is_training 1 \
  --model_id AQWan_$seq_len'_'$manifold'_'192_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom_decomposition \
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
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --window_size $window_size \
  --use_moving_window


python run.py \
  --is_training 1 \
  --model_id AQWan_$seq_len'_'$manifold_type'_'336_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom_decomposition \
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
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --window_size 4 \
  --use_moving_window

python run.py \
  --is_training 1 \
  --model_id AQWan_$seq_len'_'$manifold_type'_'720_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom_decomposition \
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
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --window_size 6 \
  --use_moving_window
