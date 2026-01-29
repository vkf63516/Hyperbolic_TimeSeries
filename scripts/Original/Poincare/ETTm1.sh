seq_len=720
train_epochs=30
patience=5
enc_in=7
manifold_type="Poincare"
data_path=ETTm1.csv
hidden_dim=256
num_basis=10 
window_size=5

#python run.py \
#  --is_training 1 \
#  --hyperbolic_weight 0.1 \
#  --hierarchy_weight 0.1 \
#  --model_id ETTm1_$seq_len'_'96_$manifold_type'_'type_exp1_Segment \
#  --model HyperbolicForecasting \
#  --data ETTm1 \
#  --root_path ./time-series-dataset/dataset/ \
#  --data_path $data_path \
#  --features M \
#  --num_basis $num_basis \
#  --label_len 0 \
#  --seq_len $seq_len \
#  --pred_len 96 \
#  --lradj "type3" \
#  --encode_dim 64 \
#  --hidden_dim $hidden_dim \
#  --batch_size 64 \
#  --learning_rate 1e-3 \
#  --train_epochs $train_epochs \
#  --use_learnable_decomposition \
#  --enc_in $enc_in \
#  --patience $patience \
#  --manifold_type $manifold_type \
#  --use_revin \
#  --use_wandb \
#  --use_segments \
#  --mstl_period 48 \
#  --fine_period 96 \
#  --coarse_period 672 \
#  --window_size $window_size
#
##python run.py \
#  --is_training 1 \
#  --hyperbolic_weight 0.1 \
#  --hierarchy_weight 0.1 \
#  --model_id ETTm1_$seq_len'_'192_$manifold_type'_'exp1_Segment \
#  --model HyperbolicForecasting \
#  --data ETTm1 \
#  --root_path ./time-series-dataset/dataset/ \
#  --data_path $data_path \
#  --features M \
#  --num_basis $num_basis \
#  --seq_len $seq_len \
#  --label_len 0 \
#  --pred_len 192 \
#  --lradj "type3" \
#  --encode_dim 64 \
#  --hidden_dim $hidden_dim \
#  --batch_size 64 \
#  --learning_rate 1e-3 \
#  --train_epochs $train_epochs \
#  --use_learnable_decomposition \
#  --enc_in $enc_in \
#  --patience $patience \
#  --manifold_type $manifold_type \
#  --use_revin \
#  --use_wandb \
#  --use_segments \
#  --mstl_period 48 \
#  --fine_period 96 \
#  --coarse_period 672 \
#  --window_size $window_size
#
#python run.py \
#  --is_training 1 \
#  --hyperbolic_weight 0.1 \
#  --hierarchy_weight 0.1 \
#  --model_id ETTm1_$seq_len'_'336_$manifold_type'_'exp1_Segment \
#  --model HyperbolicForecasting \
#  --data ETTm1 \
#  --root_path ./time-series-dataset/dataset/ \
#  --data_path $data_path \
#  --features M \
#  --num_basis $num_basis \
#  --seq_len $seq_len \
#  --label_len 0 \
#  --pred_len 336 \
#  --lradj "type3" \
#  --encode_dim 64 \
#  --hidden_dim $hidden_dim \
#  --batch_size 64 \
#  --learning_rate 1e-3 \
#  --train_epochs $train_epochs \
#  --use_learnable_decomposition \
#  --enc_in $enc_in \
#  --patience $patience \
#  --manifold_type $manifold_type \
#  --use_revin \
#  --use_wandb \
#  --use_segments \
#  --mstl_period 48 \
#  --fine_period 96 \
#  --coarse_period 672 \
#  --window_size $window_size

python run.py \
  --is_training 1 \
  --hyperbolic_weight 0.1 \
  --hierarchy_weight 0.1 \
  --model_id ETTm1_$seq_len'_'720_$manifold_type'_'exp1_segment \
  --model HyperbolicForecasting \
  --data ETTm1 \
  --root_path ./time-series-dataset/dataset/ \
  --data_path $data_path \
  --features M \
  --num_basis $num_basis \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --lradj "type3" \
  --encode_dim 64 \
  --hidden_dim $hidden_dim \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --train_epochs $train_epochs \
  --use_learnable_decomposition \
  --enc_in $enc_in \
  --patience $patience \
  --manifold_type $manifold_type \
  --use_revin \
  --use_wandb \
  --use_segments \
  --mstl_period 48 \
  --fine_period 96 \
  --coarse_period 672 \
  --window_size $window_size

