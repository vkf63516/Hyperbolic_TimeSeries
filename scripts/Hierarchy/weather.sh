window_trend=4320
window_coarse=1008
window_fine=144
seq_len=96
data_path="weather.csv"
enc_in=21
mstl_period=24
save_dir="./plots/hierarchy_weather"

python hierarchical_analysis.py \
  --root_path ./time-series-dataset/dataset/ \
  --data custom_decomposition \
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
