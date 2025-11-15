
seq_len=96
train_epochs=30
patience=6

python run.py \
  --is_training 1 \
  --model_id ETTm1_$seq_len'_'96_exp1_HScales \
  --model HyperbolicForecasting \
  --data ETTm1_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path ETTm1.csv \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 96 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 5e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in 7 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_revin \
  --use_attention_pooling

python run.py \
  --is_training 1 \
  --model_id ETTm1_$seq_len'_'192_exp1_HScales \
  --model HyperbolicForecasting \
  --data ETTm1_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path ETTm1.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in 7 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_revin \
  --use_attention_pooling


python run.py \
  --is_training 1 \
  --model_id ETTm1_$seq_len'_'336_exp1_HScales \
  --model HyperbolicForecasting \
  --data ETTm1_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path ETTm1.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 7 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_revin


python run.py \
  --is_training 1 \
  --model_id ETTm1_$seq_len'_'720_exp1_HScales \
  --model HyperbolicForecasting \
  --data ETTm1_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path ETTm1.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 7 \
  --patience 6 \
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_revin 
