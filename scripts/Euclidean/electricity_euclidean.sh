
python run.py \
  --is_training 1 \
  --model_id Electricity_96_96_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path electricity.csv \
  --features M \
  --num_basis 10 \
  --seq_len 96 \
  --pred_len 96 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 321 \
  --patience 6 \
  --manifold_type "Euclidean" \
  --use_attention_pooling \
  --use_amp

python run.py \
  --is_training 1 \
  --model_id Electricity_96_192_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path electricity.csv \
  --features M \
  --num_basis 10 \
  --seq_len 96 \
  --pred_len 192 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 321 \
  --patience 6 \
  --manifold_type "Euclidean" \
  --use_attention_pooling \
  --use_amp


python run.py \
  --is_training 1 \
  --model_id Electricity_96_336_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path electricity.csv \
  --features M \
  --num_basis 10 \
  --seq_len 96 \
  --pred_len 336 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 321 \
  --patience 6 \
  --manifold_type "Euclidean" \
  --use_attention_pooling \
  --use_amp 


python run.py \
  --is_training 1 \
  --model_id Electricity_96_720_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/ \
  --data_path electricity.csv \
  --features M \
  --num_basis 10 \
  --seq_len 96 \
  --pred_len 720 \
  --lradj "type3" \
  --embed_dim 32 \
  --hidden_dim 64 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --train_epochs 30 \
  --use_decomposition \
  --enc_in 321 \
  --patience 6 \
  --manifold_type "Euclidean" \
  --use_attention_pooling \
  --use_amp 
