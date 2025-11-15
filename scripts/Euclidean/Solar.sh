seq_len=96
train_epochs=30
patience=6

python run.py \
  --is_training 1 \
  --model_id Solar_$seq_len'_'96_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path Solar.csv \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 96 \
  --lradj "type3" \
  --embed_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-4 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in 137 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_revin 


python run.py \
  --is_training 1 \
  --model_id Solar_$seq_len'_'192_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path Solar.csv \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 192 \
  --lradj "type3" \
  --embed_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-4 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in 137 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_revin 

python run.py \
  --is_training 1 \
  --model_id Solar_$seq_len'_'336_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path Solar.csv \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 336 \
  --lradj "type3" \
  --embed_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-4 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in 137 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_revin 



python run.py \
  --is_training 1 \
  --model_id Solar_$seq_len'_'720_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path Solar.csv \
  --features M \
  --num_basis 15 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 720 \
  --lradj "type3" \
  --embed_dim 64 \
  --hidden_dim 256 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-4 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in 137 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_revin 
