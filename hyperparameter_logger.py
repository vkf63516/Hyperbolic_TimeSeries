import csv
import os
from datetime import datetime

class HyperparameterLogger: 
    def __init__(self, dataset_name, pred_len):
        self.dataset_name = dataset_name
        self.pred_len = pred_len
        self.csv_file = f"{dataset_name}_pred{pred_len}_hyperparam_results.csv"
        self.best_csv_file = f"{dataset_name}_best_hyperparams.csv"
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'hyperbolic_weight', 'hierarchy_weight', 
                               'learning_rate', 'window_size', 'pred_len', 
                               'mse', 'mae', 'setting'])
    
    def log_result(self, hyperbolic_weight, hierarchy_weight, lr, window_size, 
                   mse, mae, setting):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), hyperbolic_weight, 
                           hierarchy_weight, lr, window_size, self.pred_len,
                           mse, mae, setting])
    
    def find_best_hyperparams(self):
        """Read all results and find best hyperparameters for each pred_len"""
        results = {}
        
        # Read all CSV files for this dataset
        for pred_len in [96, 192, 336, 720]: 
            csv_file = f"{self.dataset_name}_pred{pred_len}_hyperparam_results.csv"
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    best_mse = float('inf')
                    best_params = None
                    
                    for row in reader:
                        mse = float(row['mse'])
                        if mse < best_mse:
                            best_mse = mse
                            best_params = row
                    
                    if best_params:
                        results[pred_len] = best_params
        
        # Write best parameters to summary file
        if results:
            with open(self.best_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['pred_len', 'best_hyperbolic_weight', 'best_hierarchy_weight',
                               'best_learning_rate', 'best_window_size', 'best_mse', 'best_mae'])
                
                for pred_len in sorted(results.keys()):
                    params = results[pred_len]
                    writer.writerow([pred_len, params['hyperbolic_weight'], 
                                   params['hierarchy_weight'], params['learning_rate'],
                                   params['window_size'], params['mse'], params['mae']])