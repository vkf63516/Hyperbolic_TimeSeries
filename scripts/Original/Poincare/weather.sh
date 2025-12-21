seq_len=96
train_epochs=30
patience=6
enc_in=21
manifold_type="Poincare"
data_path=weather.csv

python run.py \
  --is_training 1 \
  --model_id Weather_$seq_len'_'$manifold_type'_'96_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 96 \
  --lradj "type3" \
  --encode_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments


python run.py \
  --is_training 1 \
  --model_id Weather_$seq_len'_'$manifold'_'192_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 192 \
  --lradj "type3" \
  --encode_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments

python run.py \
  --is_training 1 \
  --model_id Weather_$seq_len'_'$manifold_type'_'336_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 336 \
  --lradj "type3" \
  --encode_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments

python run.py \
  --is_training 1 \
  --model_id Weather_$seq_len'_'$manifold_type'_'720_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 720 \
  --lradj "type3" \
  --encode_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments
