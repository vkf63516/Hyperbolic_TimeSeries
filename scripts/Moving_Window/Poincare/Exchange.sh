seq_len=720
train_epochs=30
patience=6
enc_in=8
manifold_type="Poincare"
num_basis=6
window_size=2
data_path=Exchange.csv

python run.py \
  --is_training 1 \
  --model_id Exchange_$seq_len'_'$manifold_type'_'96_exp1_Segment \
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
  --encode_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --mstl_period 4 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --use_moving_window


python run.py \
  --is_training 1 \
  --model_id Exchange_$seq_len'_'$manifold'_'192_exp1_Segment \
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
  --mstl_period 4 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --use_moving_window

python run.py \
  --is_training 1 \
  --model_id Exchange_$seq_len'_'$manifold_type'_'336_exp1_Segment \
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
  --mstl_period 4 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --use_moving_window

python run.py \
  --is_training 1 \
  --model_id Exchange_$seq_len'_'$manifold_type'_'720_exp1_Segment \
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
  --mstl_period 4 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --use_moving_window
