seq_len=720
train_epochs=30
patience=5
enc_in=321
encode_dim=64
manifold_type="Poincare"
data_path=electricity.csv
window_size=5
python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.05 \
  --hierarchy_weight 0.001 \
  --model_id Electricity_$seq_len'_'$manifold_type'_'96_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 96 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 16 \
  --learning_rate 5e-3 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --use_multi_horizon \
  --window_size $window_size


python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.05 \
  --hierarchy_weight 0.01 \
  --model_id Electricity_$seq_len'_'$manifold'_'192_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 192 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 16 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --use_multi_horizon \
  --window_size 4

python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.05 \
  --hierarchy_weight 0.01 \
  --model_id Electricity_$seq_len'_'$manifold_type'_'336_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 336 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 16 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --use_multi_horizon \
  --window_size $window_size

python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.1 \
  --hierarchy_weight 0.1 \
  --model_id Electricity_$seq_len'_'$manifold_type'_'720_exp1_Segment \
  --model HyperbolicForecasting \
  --data custom \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis 10 \
  --label_len 0 \
  --seq_len $seq_len \
  --pred_len 720 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim 256 \
  --batch_size 16 \
  --use_wandb \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_segments \
  --use_multi_horizon \
  --window_size $window_size
