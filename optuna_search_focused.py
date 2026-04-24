"""
Focused Optuna Search - Core Hyperparameters Only
User: vkf63516
Date: 2025-11-13 21:08:39 UTC

Searches only: encode_dim, hidden_dim, learning_rate, batch_size
"""

import optuna
import subprocess
import json
import os
from datetime import datetime
import pandas as pd
import argparse

def parse_result_file(result_file='result.txt'):
    """Parse latest result from result.txt"""
    if not os.path.exists(result_file):
        print("Cant find file")
        return None
    
    with open(result_file, 'r') as f:
        lines = f.readlines()
    
    non_empty = [line.strip() for line in lines if line.strip()]
    print(len(non_empty))
    
    if len(non_empty) >= 2:
        print(non_empty[-3])
        try:
            metrics_line = non_empty[-3]
            print(metrics_line)
            if 'mse:' in metrics_line:
                parts = metrics_line.split(',')
                mse = float(parts[0].split(':')[1])
                mae = float(parts[1].split(':')[1])
                rmse = float(parts[2].split(':')[1])
                return {'mse': mse, 'mae': mae, 'rmse': rmse}
        except Exception as e:
            print(f"Warning: Could not parse result - {e}")
    return None


def objective(trial, pred_len, manifold_type='Euclidean'):
    """
    Focused optimization - only core hyperparameters.
    
    FIXED settings (not searched):
    """
    
    # ONLY sample these 4 core hyperparameters
    encode_dim = trial.suggest_categorical('encode_dim', [32, 64])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    
    # Optional: Add curvature for hyperbolic manifolds
    if manifold_type in ['Poincare']:
        curvature = 1.0
    else:
        curvature = 0.0
    
    # Constraint: hidden_dim should be >= encode_dim
    if hidden_dim < encode_dim:
        hidden_dim = encode_dim * 2
    
    config = {
        'encode_dim': encode_dim,
        'hidden_dim': hidden_dim,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'curvature': curvature,
        # FIXED (not searched)
        'manifold_type': manifold_type,
    }
    
    # Build command
    cmd = [
        'python', 'run.py',
        '--is_training', '1',
        '--model', 'HyperbolicForecasting',
        '--model_id', f"ETTh1_{manifold_type}_{pred_len}_optuna_trial{trial.number}",
        '--data', 'ETTh1_decomposition',
        '--root_path', './time-series-dataset/dataset/',
        '--data_path', 'ETTh1.csv',
        '--features', 'M',
        '--seq_len', '96',
        '--pred_len', str(pred_len),
        '--encode_dim', str(encode_dim),
        '--hidden_dim', str(hidden_dim),
        '--learning_rate', f'{learning_rate:.6f}',
        '--batch_size', str(batch_size),
        '--curvature', f'{curvature:.3f}',
        '--manifold_type', manifold_type,
        '--train_epochs', '30',
        '--patience', '5',
        '--use_decomposition',
        '--enc_in', '7',
        '--num_basis', '10',
        '--use_revin',
        '--use_segments', 
        '--result_file', 'result_optuna.txt' 
        '--use_moving_window'
        # FIXED settings
    ]
    
    print("\n" + "="*80)
    print(f"[Trial {trial.number}] {manifold_type} manifold, pred_len={pred_len}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("SEARCHING: encode_dim, hidden_dim, learning_rate, batch_size")
    print(json.dumps({k: v for k, v in config.items() if k in ['encode_dim', 'hidden_dim', 'learning_rate', 'batch_size']}, indent=2))
    print("="*80)
    
    # Run experiment
    start_time = datetime.now()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Parse result
        metrics = parse_result_file()
        
        if metrics or result.returncode == 0:
            mse = metrics['mse']
            print(f"✅ Success! MSE: {mse:.6f}, MAE: {metrics['mae']:.6f}, Duration: {duration/60:.1f}min")
            
            # Log additional metrics
            trial.set_user_attr('mae', metrics['mae'])
            trial.set_user_attr('rmse', metrics['rmse'])
            
            return mse
        else:
            print(f"❌ Failed")
            return float('inf')
            
    except subprocess.TimeoutExpired:
        print("⚠️  TIMEOUT")
        return float('inf')
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description='Focused Optuna Search (Core Hyperparameters Only)')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of trials per (manifold, horizon) pair')
    parser.add_argument('--pred_lens', type=int, nargs='+', default=[96, 192, 336, 720],
                        help='Prediction horizons to optimize')
    parser.add_argument('--manifold_types', type=str, nargs='+', 
                        default=['Poincare'],
                        choices=['Euclidean', 'Lorentzian', 'Poincare'],
                        help='Manifold types to test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_focused.db',
                        help='Database for persistent storage')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing database')
    parser.add_argument('--output', type=str, default='optuna_focused_results.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 FOCUSED OPTUNA SEARCH")
    print("   Core Hyperparameters Only")
    print("="*80)
    print(f"User: vkf63516")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"\nSEARCHING:")
    print(f"  - encode_dim: [32, 48, 64, 80, 96]")
    print(f"  - hidden_dim: [64, 128, 256]")
    print(f"  - learning_rate: [1e-5, 1e-3] (log uniform)")
    print(f"  - batch_size: [16, 32, 64]")
    print(f"\nFIXED:")
    print(f"\nManifold types: {args.manifold_types}")
    print(f"Storage: {args.storage}")
    print(f"Resume mode: {args.resume}")
    print(f"Trials per (manifold, horizon): {args.n_trials}")
    print(f"Prediction horizons: {args.pred_lens}")
    total_combos = len(args.manifold_types) * len(args.pred_lens)
    print(f"\nTotal combinations: {total_combos}")
    print(f"Total trials: {total_combos * args.n_trials}")
    print(f"Estimated time: {total_combos * args.n_trials * 15 / 60:.1f} hours")
    print("="*80 + "\n")
    
    all_results = []
    
    for manifold_type in args.manifold_types:
        for pred_len in args.pred_lens:
            study_name = f'{manifold_type}_pred{pred_len}_focused_v1'
            
            print(f"\n{'='*80}")
            print(f"🎯 OPTIMIZING: {manifold_type} manifold, pred_len={pred_len}")
            print(f"Study name: {study_name}")
            print(f"{'='*80}\n")
            
            # Create or load study
            study = optuna.create_study(
                study_name=study_name,
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                pruner=optuna.pruners.MedianPruner(),
                storage=args.storage,
                load_if_exists=args.resume
            )
            
            n_existing = len(study.trials)
            print(f"📊 Existing trials: {n_existing}")
            if n_existing > 0:
                print(f"✅ Reusing {n_existing} previous trials!")
                print(f"🏆 Best MSE so far: {study.best_value:.6f}")
            
            print(f"🔄 Running {args.n_trials} NEW trials...\n")
            
            # Optimize
            study.optimize(
                lambda trial: objective(trial, pred_len, manifold_type),
                n_trials=args.n_trials,
                show_progress_bar=True
            )
            
            # Print results
            print(f"\n{'='*80}")
            print(f"✅ COMPLETE: {manifold_type}, pred_len={pred_len}")
            print(f"{'='*80}")
            print(f"Total trials: {len(study.trials)}")
            print(f"Best MSE: {study.best_value:.6f}")
            print(f"\nBest parameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            
            # Save results
            for trial in study.trials:
                if trial.value != float('inf'):
                    result_entry = {
                        'manifold_type': manifold_type,
                        'pred_len': pred_len,
                        'trial_number': trial.number,
                        'mse': trial.value,
                        'mae': trial.user_attrs.get('mae', None),
                        'rmse': trial.user_attrs.get('rmse', None),
                        'duration_seconds': trial.user_attrs.get('duration_seconds', None),
                        **trial.params,
                        # Add fixed params for reference
                    }
                    all_results.append(result_entry)
            
            df = pd.DataFrame(all_results)
            df.to_csv(args.output, index=False)
            print(f"\n💾 Results saved to {args.output}")
            
            # Plot optimization history
            try:
                import matplotlib.pyplot as plt
                
                fig = optuna.visualization.matplotlib.plot_optimization_history(study)
                plt.savefig(f'optuna_focused_history_{manifold_type}_{pred_len}.png', dpi=150, bbox_inches='tight')
                print(f"📊 Plot saved to optuna_focused_history_{manifold_type}_{pred_len}.png")
                plt.close()
            except:
                pass
    
    # Final summary
    print("\n" + "="*80)
    print("🏆 FINAL SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    for manifold_type in args.manifold_types:
        print(f"\n{'='*60}")
        print(f"{manifold_type.upper()} MANIFOLD")
        print(f"{'='*60}")
        
        df_m = df.loc[df['manifold_type'] == manifold_type]
        
        for pred_len in sorted(df_m['pred_len'].unique()):
            df_h = df_m.loc[df_m['pred_len'] == pred_len]
            best = df_h.loc[df_h['mse'].idxmin()]
            print(f"\npred_len={pred_len}:")
            print(f"  Best MSE: {best['mse']:.6f}, MAE: {best['mae']:.6f}")
            print(f"  encode_dim={best['encode_dim']}, hidden_dim={best['hidden_dim']}")
            print(f"  lr={best['learning_rate']:.6f}, batch={best['batch_size']}")
            if 'curvature' in best:
                print(f"  curvature={best['curvature']:.3f}")
    
    print(f"\n💾 All results: {args.output}")
    print(f"💾 Database: {args.storage}")
    print(f"⏱️  Total time: {df['duration_seconds'].sum() / 3600:.1f} hours")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()