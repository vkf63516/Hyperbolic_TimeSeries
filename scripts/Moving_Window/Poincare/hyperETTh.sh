# Fixed parameters
seq_len=720
train_epochs=30
patience=5
enc_in=7
manifold_type="Poincare"
data_path=ETTh1.csv
hidden_dim=256
num_basis=10

# Hyperparameter grids
hyperbolic_weights=(0.05 0.1 0.2)
hierarchy_weights=(0.0001 0.0005 0.1)
learning_rates=(1e-2 1e-3)
window_sizes=(2 4 5 6)
pred_lens=(96 192 336 720)

# Period parameters for Temp
fine_period=24
coarse_period=168

echo "Starting hyperparameter tuning for ETTh1 dataset"
echo "Total combinations: $((${#hyperbolic_weights[@]} * ${#hierarchy_weights[@]} * ${#learning_rates[@]} * ${#window_sizes[@]} * ${#pred_lens[@]}))"

for pred_len in "${pred_lens[@]}"; do
    echo "========================================="
    echo "Pred Length: $pred_len"
    echo "========================================="
    
    for hyp_weight in "${hyperbolic_weights[@]}"; do
        for hier_weight in "${hierarchy_weights[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for win_size in "${window_sizes[@]}"; do
                    echo "Running: hyp_w=$hyp_weight, hier_w=$hier_weight, lr=$lr, win=$win_size"
                    
                    python -u run.py \
                      --is_training 1 \
                      --hyperbolic_weight $hyp_weight \
                      --hierarchy_weight $hier_weight \
                      --model_id ETTh1_${seq_len}_${pred_len}_${manifold_type}_HP_h${hyp_weight}_hi${hier_weight}_lr${lr}_w${win_size} \
                      --model HyperbolicForecasting \
                      --data ETTh1 \
                      --root_path /mmfs1/project/mx6/sp3463/HTS-v/dataset/ \
                      --data_path $data_path \
                      --features M \
                      --num_basis $num_basis \
                      --label_len 0 \
                      --seq_len $seq_len \
                      --pred_len $pred_len \
                      --lradj "type3" \
                      --encode_dim 64 \
                      --hidden_dim $hidden_dim \
                      --batch_size 32 \
                      --learning_rate $lr \
                      --train_epochs $train_epochs \
                      --use_learnable_decomposition \
                      --enc_in $enc_in \
                      --patience $patience \
                      --manifold_type $manifold_type \
                      --use_revin \
                      --use_wandb \
                      --use_segments \
                      --fine_period $fine_period \
                      --coarse_period $coarse_period \
                      --window_size $win_size \
                      --use_moving_window \
                      --result_file ETTh1_pred${pred_len}_hyperparam_results.csv
                    
                    echo "Completed: hyp_w=$hyp_weight, hier_w=$hier_weight, lr=$lr, win=$win_size"
                    
                    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
                    sleep 5
                done
            done
        done
    done
done

echo "Hyperparameter tuning completed at $(date)"

# Find best hyperparameters
python -c "
import csv
import os

dataset_name = 'ETTh1'
pred_lens = [96, 192, 336, 720]

results = {}
for pred_len in pred_lens: 
    csv_file = f'{dataset_name}_pred{pred_len}_hyperparam_results.csv'
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            best_mse = float('inf')
            best_params = None
            
            for row in reader:
                try:
                    mse = float(row.get('mse', float('inf')))
                    if mse < best_mse:
                        best_mse = mse
                        best_params = row
                except: 
                    continue
            
            if best_params:
                results[pred_len] = best_params

if results:
    with open(f'{dataset_name}_best_hyperparams.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pred_len', 'best_hyperbolic_weight', 'best_hierarchy_weight',
                       'best_learning_rate', 'best_window_size', 'best_mse', 'best_mae'])
        
        for pred_len in sorted(results.keys()):
            params = results[pred_len]
            writer.writerow([pred_len, params.get('hyperbolic_weight', ''), 
                           params.get('hierarchy_weight', ''), params.get('learning_rate', ''),
                           params.get('window_size', ''), params.get('mse', ''), params.get('mae', '')])
    print(f'Best hyperparameters saved to {dataset_name}_best_hyperparams.csv')
"

echo "Best hyperparameters extracted"
