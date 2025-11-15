seq_len=96
patience=5
batch_size=128
lr=0.001
python run.py \
  --is_training 1 \
  --model_id Weather_$seq_len'_'96_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path weather.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --learning_rate $lr \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 21 \
  --use_wandb \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_revin

python run.py \
  --is_training 1 \
  --model_id Weather_$seq_len'_'192_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path weather.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --learning_rate $lr \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 21 \
  --use_wandb \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_revin

python run.py \
  --is_training 1 \
  --model_id Weather_$seq_len'_'336_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path weather.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --learning_rate $lr \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 21 \
  --use_wandb \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_revin

python run.py \
  --is_training 1 \
  --model_id Weather_$seq_len'_'720_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path weather.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size  \
  --learning_rate $lr \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 21 \
  --use_wandb \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_revin