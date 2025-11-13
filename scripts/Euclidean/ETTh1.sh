
seq_len=96
train_epochs=10
patience=10

python run.py \
  --is_training 1 \
  --model_id ETTh1_$seq_len'_'96_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --pred_len 96 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 16 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in 7 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_attention_pooling \
  --use_gpu \
  --use_amp

python run.py \
  --is_training 1 \
  --model_id ETTh1_$seq_len'_'192_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --pred_len 192 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 16 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_decomposition \
  --enc_in 7 \
  --patience $patience \
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_attention_pooling \
  --use_gpu \
  --use_amp


python run.py \
  --is_training 1 \
  --model_id ETTh1_$seq_len'_'336_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --pred_len 336 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 16 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 7 \
  --patience 6 \
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_attention_pooling \
  --use_gpu \
  --use_amp 


python run.py \
  --is_training 1 \
  --model_id ETTh1_$seq_len'_'720_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --num_basis 10 \
  --seq_len $seq_len \
  --pred_len 720 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 16 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 7 \
  --patience 6 \
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_attention_pooling \
  --use_gpu \
  --use_amp 
