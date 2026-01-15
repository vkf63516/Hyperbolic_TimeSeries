window_trend=720
window_coarse=168
window_fine=24
seq_len=96
data_path="traffic.csv"
enc_in=862
mstl_period=24
save_dir="./plots/hierarchy_traffic"

python hierarchical_analysis.py \
  --root_path ./time-series-dataset/dataset/ \
  --data custom\
  --features "M" \
  --enc_in $enc_in \
  --seq_len $seq_len \
  --window_trend $window_trend \
  --window_coarse $window_coarse \
  --window_fine $window_fine \
  --data_path $data_path \
  --label_len 0 \
  --pred_len 96 \
  --save_dir $save_dir
