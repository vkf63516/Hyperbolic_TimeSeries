python run.py \
  --is_training 1 \
  --model_id Weather_96_exp1_No_HScales \
  --model HyperbolicMambaForecasting \
  --data custom_decomposition \
  --root_path ./time-series-dataset/dataset/weather/ \
  --data_path weather.csv \
  --features MS \
  --seq_len 720 \
  --pred_len 96 \
  --embed_dim 32 \
  --hidden_dim 128 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --use_decomposition \

