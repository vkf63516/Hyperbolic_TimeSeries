python run.py \
  --is_training 1 \
  --model_id Weather_336_96_exp1_HScales \
  --model HyperbolicForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/weather/ \
  --data_path weather.csv \
  --features MS \
  --seq_len 336 \
  --pred_len 96 \
  --embed_dim 32 \
  --hidden_dim 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 50 \
  --use_decomposition \
  --enc_in 21 \
  --use_tensorboard \
  --patience 6
  --manifold_type "Euclidean" \
  --use_hierarchy \
  --use_amp


