"""
Enhanced evaluation script with two key functions:
1. Recover cross-validation results from already trained models
2. Evaluate a trained model on a new dataset for cross-dataset comparison
"""
import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import create_model
from evaluation import evaluate_model
from dataset import MovementDataset
from dct_utils import build_dct_matrix, build_idct_matrix, batch_dct_transform, batch_idct_transform
from config import T_OBS, T_PRED

from training_utils import create_cross_validation_splits
import pickle

def convert_numpy_to_python(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj


def tensor_to_python(value):
    """Convert tensor values to native Python types."""
    if isinstance(value, torch.Tensor):
        try:
            return value.item()  # For scalar tensors
        except ValueError:
            return str(value)    # For non-scalar tensors
    elif isinstance(value, (list, tuple)):
        return [tensor_to_python(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): tensor_to_python(v) for k, v in value.items()}
    return value

def recover_cv_results(data_path, model_path_template, n_folds=5, device=None):
    """Recover cross-validation results from already trained models and generate thesis figures/tables."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataset and create CV splits
    from training_utils import create_cross_validation_splits
    import pickle
    
    # Try to load saved CV splits if they exist
    cv_splits_path = "../evaluations/cross_validation/cv_splits.pkl"
    if os.path.exists(cv_splits_path):
        print(f"Loading saved CV splits from {cv_splits_path}")
        with open(cv_splits_path, 'rb') as f:
            cv_splits = pickle.load(f)
    else:
        # Create dataset and regenerate splits
        print("Regenerating CV splits...")
        dataset = MovementDataset(
            h5_path=data_path,
            T_obs=T_OBS,
            T_pred=T_PRED
        )
        cv_splits = create_cross_validation_splits(dataset, n_folds=n_folds, by='subject_id')
        
        # Save the splits for reproducibility
        os.makedirs(os.path.dirname(cv_splits_path), exist_ok=True)
        with open(cv_splits_path, 'wb') as f:
            pickle.dump(cv_splits, f)
    
    # Define target horizons to match thesis (in ms)
    target_horizons = [80, 160, 320, 400, 600, 1000]
    
    # Store results across folds
    all_fold_results = []
    best_fold_score = float('inf')
    best_fold_idx = 0
    
    # Loop through all folds
    for fold_idx, (fold_train_keys, fold_test_keys) in enumerate(cv_splits):
        print(f"\n===== Processing Fold {fold_idx+1}/{n_folds} =====")
        
        # Create test dataset for this fold
        fold_test_dataset = MovementDataset(
            h5_path=data_path,
            T_obs=T_OBS,
            T_pred=T_PRED,
            prefiltered=True,
            filtered_keys=fold_test_keys,
            dataset_type=f"test_fold_{fold_idx+1}"
        )
        
        # Load model for this fold
        fold_model = create_model(
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            device=device
        )
        
        # Construct model path
        fold_model_path = model_path_template.replace('.pth', f'_fold_{fold_idx+1}.pth')
        
        # Skip folds where model doesn't exist
        if not os.path.exists(fold_model_path):
            print(f"WARNING: Model for fold {fold_idx+1} not found at {fold_model_path}")
            continue
            
        # Load model weights
        fold_model.load_state_dict(torch.load(fold_model_path, map_location=device))
        print(f"Loaded model from {fold_model_path}")
        
        # Evaluate model
        fold_dir = f"../evaluations/fold_{fold_idx+1}"
        os.makedirs(fold_dir, exist_ok=True)
        
        fold_results = evaluate_model(
            model=fold_model,
            dataset=fold_test_dataset,
            build_dct_matrix=build_dct_matrix,
            build_idct_matrix=build_idct_matrix,
            batch_dct_transform=batch_dct_transform,
            batch_idct_transform=batch_idct_transform,
            device=device,
            batch_size=8,
            output_dir=fold_dir
        )
        
        # Store results
        all_fold_results.append(fold_results)
        
        # Check if this is the best fold
        avg_error = np.mean([v for v in fold_results['overall'].values()])
        if avg_error < best_fold_score:
            best_fold_score = avg_error
            best_fold_idx = fold_idx
    
    # Now aggregate the results and create thesis outputs
    if all_fold_results:
        # Create output directory
        output_dir = "../evaluations/cross_validation"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Compute average metrics across folds
        avg_results = {'overall': {}, 'by_table': {}, 'by_movement': {}}
        
        # Average overall metrics
        for ms in target_horizons:
            ms_str = str(ms)
            values = [fold['overall'].get(ms_str, 0) for fold in all_fold_results 
                     if ms_str in fold['overall']]
            if values:
                avg_results['overall'][ms_str] = float(np.mean(values))
        
        # 2. Generate overall performance table (Table ref:overall_performance)
        with open(f'{output_dir}/overall_performance_table.csv', 'w') as f:
            f.write('Time Horizon,Mean Absolute Error (radians)\n')
            for ms in target_horizons:
                ms_str = str(ms)
                if ms_str in avg_results['overall']:
                    f.write(f'{ms},{avg_results["overall"][ms_str]:.4f}\n')
        
        # 3. Generate error progression figure (Fig ref:error_progression)
        plt.figure(figsize=(10, 6))
        x = []
        y = []
        for ms in target_horizons:
            ms_str = str(ms)
            if ms_str in avg_results['overall']:
                x.append(ms)
                y.append(avg_results['overall'][ms_str])
        
        plt.plot(x, y, 'o-', linewidth=2)
        plt.xlabel('Prediction Horizon (ms)')
        plt.ylabel('Mean Absolute Error (radians)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title('Error Progression Over Prediction Time Horizon')
        plt.savefig(f'{output_dir}/error_progression.png', dpi=300)
        plt.close()
        
        # 4. Compute movement-specific results
        movement_results = {}
        for fold_result in all_fold_results:
            if 'by_movement' in fold_result:
                for movement, metrics in fold_result['by_movement'].items():
                    # Convert movement key to string (to avoid tensor keys)
                    movement_str = str(movement)
                    if movement_str not in movement_results:
                        movement_results[movement_str] = {str(ms): [] for ms in target_horizons}
                    
                    for ms_str, error in metrics.items():
                        if ms_str in movement_results[movement_str]:
                            # Explicitly convert error values to float to handle tensors
                            if isinstance(error, list):
                                # If it's a list, convert each element
                                converted_errors = []
                                for e in error:
                                    try:
                                        converted_errors.append(float(tensor_to_python(e)))
                                    except (ValueError, TypeError):
                                        # Skip values that can't be converted to float
                                        pass
                                movement_results[movement_str][ms_str].extend(converted_errors)
                            else:
                                # Single value
                                try:
                                    movement_results[movement_str][ms_str].append(float(tensor_to_python(error)))
                                except (ValueError, TypeError):
                                    # Skip if can't convert to float
                                    pass
                
        # Average movement results across folds
        movement_avg = {}
        for movement, metrics in movement_results.items():
            movement_avg[movement] = {}
            for ms_str, errors in metrics.items():
                if errors:  # Only include if we have data
                    try:
                        # Ensure all values are scalar
                        scalar_errors = []
                        for e in errors:
                            try:
                                scalar_errors.append(float(e))
                            except (TypeError, ValueError):
                                print(f"Warning: Skipping non-scalar error value for {movement} at {ms_str}ms")
                        
                        if scalar_errors:
                            movement_avg[movement][ms_str] = float(np.mean(scalar_errors))
                        else:
                            print(f"Warning: No valid error values for {movement} at {ms_str}ms")
                            movement_avg[movement][ms_str] = float('nan')
                    except Exception as e:
                        print(f"Error processing {movement} at {ms_str}ms: {e}")
                        movement_avg[movement][ms_str] = float('nan')
        
        # 5. Generate best/worst movements analysis (for thesis text)
        time_horizon = '400'  # For analysis at 400ms in thesis
        movement_at_horizon = []
        
        for movement, metrics in movement_avg.items():
            if time_horizon in metrics:
                movement_at_horizon.append((movement, metrics[time_horizon]))
        
        # Sort by performance
        movement_at_horizon.sort(key=lambda x: x[1])
        
        with open(f'{output_dir}/best_worst_movements.txt', 'w') as f:
            f.write("Best performing movements:\n")
            for i, (movement, error) in enumerate(movement_at_horizon[:5]):
                f.write(f"{i+1}. {movement}: {error:.4f} radians\n")
            
            f.write("\nWorst performing movements:\n")
            for i, (movement, error) in enumerate(movement_at_horizon[-5:]):
                f.write(f"{i+1}. {movement}: {error:.4f} radians\n")
        
        # 6. Generate movement comparison visualization (Fig ref:movement_comparison)
        plt.figure(figsize=(14, 8))
        
        # Select top 5 best and worst movements for clarity
        movements_to_plot = [m for m, _ in movement_at_horizon[:5] + movement_at_horizon[-5:]]
        
        for movement in movements_to_plot:
            x = []
            y = []
            for ms in target_horizons:
                ms_str = str(ms)
                if ms_str in movement_avg[movement]:
                    x.append(ms)
                    y.append(movement_avg[movement][ms_str])
            
            plt.plot(x, y, 'o-', label=movement)
        
        plt.xlabel('Prediction Horizon (ms)')
        plt.ylabel('Mean Absolute Error (radians)')
        plt.title('Performance Comparison Across Movement Types')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(f'{output_dir}/movement_comparison.png', dpi=300)
        plt.close()
        
        # 7. Joint group analysis - this is handled by the evaluate_model function
        # for the best fold, but we'll also aggregate across folds if available
        
        # 8. Combine all joint group results if available
        joint_group_files = [f"../evaluations/fold_{i+1}/joint_group_analysis.csv" 
                           for i in range(n_folds)]
        
        joint_group_data = {}
        
        for file_path in joint_group_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    next(f)  # Skip header
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            group = parts[0]
                            error = float(parts[1])
                            
                            if group not in joint_group_data:
                                joint_group_data[group] = []
                            
                            joint_group_data[group].append(error)
        
        # Generate averaged joint group table
        if joint_group_data:
            with open(f'{output_dir}/joint_group_analysis.csv', 'w') as f:
                f.write('Joint Group,Mean Error,Standard Deviation,Min Error,Max Error\n')
                
                for group, errors in joint_group_data.items():
                    if errors:
                        f.write(f'{group},{np.mean(errors):.4f},{np.std(errors):.4f},'
                                f'{np.min(errors):.4f},{np.max(errors):.4f}\n')
        
        # 9. Create JSON data with NumPy conversion
        json_data = {
            'average': avg_results,
            'per_fold': all_fold_results,
            'best_fold': {
                'fold_score': best_fold_score,
                'fold_idx': best_fold_idx
            }
        }
        
        # Convert NumPy types to Python native types
        json_data = convert_numpy_to_python(json_data)
        
        # Save results
        with open(f'{output_dir}/aggregated_results.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print("\n===== Cross-Validation Results =====")
        print("Average error across all folds:")
        for ms, value in avg_results['overall'].items():
            print(f"  {ms}ms: {value:.4f}")
        
        print(f"\nBest fold: {best_fold_idx+1} with score {best_fold_score:.4f}")
        
        # Store the best model path for further use
        best_model_path = model_path_template.replace('.pth', f'_fold_{best_fold_idx+1}.pth')
        
    else:
        print("No fold results collected.")
        best_fold_idx = 0
        best_fold_score = float('inf')
        best_model_path = None
    
    return best_fold_idx, best_fold_score, best_model_path

def evaluate_cross_dataset(trained_model_path, original_results_path, new_dataset_path, output_dir="../evaluations/cross_dataset", device=None):
    """
    Evaluate a trained model on a new dataset and compare with original results.
    
    Args:
        trained_model_path: Path to the trained model (.pth file)
        original_results_path: Path to the original evaluation results (.json file)
        new_dataset_path: Path to the new dataset (.h5 file)
        output_dir: Directory to save evaluation outputs
        device: Device to run evaluation on
        
    Returns:
        dict: Comparison results between original and new dataset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {trained_model_path}")
    model = create_model(
        d_model=128,
        nhead=8, 
        num_layers=3,
        dim_feedforward=512,
        device=device
    )
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.eval()
    
    # Load original results for comparison
    print(f"Loading original evaluation results from {original_results_path}")
    with open(original_results_path, 'r') as f:
        original_results = json.load(f)
    
    # Create dataset from new data
    print(f"Loading new dataset from {new_dataset_path}")
    new_dataset = MovementDataset(
        h5_path=new_dataset_path,
        T_obs=T_OBS,
        T_pred=T_PRED,
        dataset_type="new_dataset"
    )
    print(f"New dataset contains {len(new_dataset)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate on new dataset
    print("Evaluating model on new dataset...")
    new_results = evaluate_model(
        model=model,
        dataset=new_dataset,
        build_dct_matrix=build_dct_matrix,
        build_idct_matrix=build_idct_matrix,
        batch_dct_transform=batch_dct_transform,
        batch_idct_transform=batch_idct_transform,
        device=device,
        batch_size=8,
        output_dir=output_dir
    )
    
    # Compare results
    if 'average' in original_results:
        # Cross-validation results format
        original_metrics = original_results['average']['overall']
    else:
        # Single evaluation results format
        original_metrics = original_results['overall']
    
    new_metrics = new_results['overall']
    
    # Create comparison
    comparison = {
        'original_dataset': original_metrics,
        'new_dataset': new_metrics,
        'difference': {
            ms: new_metrics[ms] - original_metrics[ms] 
            for ms in original_metrics if ms in new_metrics
        },
        'ratio': {
            ms: new_metrics[ms] / original_metrics[ms] if original_metrics[ms] > 0 else float('inf')
            for ms in original_metrics if ms in new_metrics
        }
    }
    
    # Save comparison results
    comparison_path = os.path.join(output_dir, "comparison_results.json")
    with open(comparison_path, 'w') as f:
        json.dump(convert_numpy_to_python(comparison), f, indent=2)
    
    # Generate a comparison table
    print("\n===== Cross-Dataset Evaluation Results =====")
    print("Time    | Original | New Dataset | Difference | Ratio")
    print("--------|----------|-------------|------------|-------")
    for ms in sorted([int(ms) for ms in original_metrics.keys()]):
        ms_str = str(ms)
        if ms_str in original_metrics and ms_str in new_metrics:
            orig_val = original_metrics[ms_str]
            new_val = new_metrics[ms_str]
            diff = new_val - orig_val
            ratio = new_val / orig_val if orig_val > 0 else float('inf')
            print(f"{ms:7d}ms | {orig_val:.4f}   | {new_val:.4f}      | {diff:+.4f}     | {ratio:.2f}x")
    
    # Generate comparison plot
    try:
        # Extract data more carefully
        orig_ts = []
        orig_vals = []
        new_ts = []
        new_vals = []
        
        # Match keys between datasets
        for key in original_metrics.keys():
            if key in new_metrics:
                try:
                    ts = int(key.rstrip('ms'))  # Handle both "80" and "80ms" formats
                    orig_ts.append(ts)
                    orig_vals.append(original_metrics[key])
                    new_ts.append(ts)
                    new_vals.append(new_metrics[key])
                except (ValueError, TypeError):
                    print(f"Skipping non-numeric key: {key}")
        
        # Sort by timestamp
        sorted_indices = np.argsort(orig_ts)
        orig_ts = [orig_ts[i] for i in sorted_indices]
        orig_vals = [orig_vals[i] for i in sorted_indices]
        new_ts = [new_ts[i] for i in sorted_indices]
        new_vals = [new_vals[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(orig_ts, orig_vals, 'o-', label='Original Dataset')
        plt.plot(new_ts, new_vals, 's-', label='New Dataset')
        plt.xlabel('Time Horizon (ms)')
        plt.ylabel('Mean Angle Error')
        plt.title('Model Performance: Original vs. New Dataset')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(output_dir, "cross_dataset_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"\nComparison plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate comparison plot: {str(e)}")
    
    print(f"\nDetailed comparison results saved to {comparison_path}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced evaluation script")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Cross-validation recovery parser
    cv_parser = subparsers.add_parser("cv", help="Recover cross-validation results")
    cv_parser.add_argument('--data', type=str, default="../datasets/shadow_dataset.h5", 
                          help="Path to the dataset")
    cv_parser.add_argument('--model-path', type=str, default="../models/best_model.pth", 
                          help="Base path for models (will add _fold_X)")
    cv_parser.add_argument('--n-folds', type=int, default=5, 
                          help="Number of folds")
    
    # Cross-dataset evaluation parser
    cross_dataset_parser = subparsers.add_parser("cross-dataset", help="Evaluate on a new dataset")
    cross_dataset_parser.add_argument('--model', type=str, required=True, 
                                    help="Path to trained model")
    cross_dataset_parser.add_argument('--original-results', type=str, required=True, 
                                    help="Path to original evaluation results")
    cross_dataset_parser.add_argument('--new-data', type=str, required=True, 
                                    help="Path to new dataset")
    cross_dataset_parser.add_argument('--output-dir', type=str, default="../evaluations/cross_dataset", 
                                    help="Directory to save outputs")
    
    # Automated cross-validation and cross-dataset evaluation
    auto_parser = subparsers.add_parser("auto", help="Run CV recovery and then cross-dataset evaluation with best model")
    auto_parser.add_argument('--data', type=str, default="../datasets/shadow_dataset.h5", 
                          help="Path to the original dataset")
    auto_parser.add_argument('--model-path', type=str, default="../models/best_model.pth", 
                          help="Base path for models")
    auto_parser.add_argument('--n-folds', type=int, default=5, 
                          help="Number of folds")
    auto_parser.add_argument('--new-data', type=str, required=True, 
                          help="Path to new dataset")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.command == "cv":
        # Run cross-validation recovery
        recover_cv_results(args.data, args.model_path, args.n_folds, device)
    
    elif args.command == "cross-dataset":
        # Run cross-dataset evaluation
        evaluate_cross_dataset(args.model, args.original_results, args.new_data, args.output_dir, device)
    
    elif args.command == "auto":
        # Run CV recovery first
        print("Step 1: Recovering cross-validation results...")
        best_fold_idx, best_fold_score, best_model_path = recover_cv_results(
            args.data, args.model_path, args.n_folds, device
        )
        
        # Path to aggregated results
        cv_results_path = "../evaluations/cross_validation/aggregated_results.json"
        
        # Then run cross-dataset evaluation with the best model
        print(f"\nStep 2: Evaluating best model (fold {best_fold_idx+1}) on new dataset...")
        evaluate_cross_dataset(
            trained_model_path=best_model_path,
            original_results_path=cv_results_path,
            new_dataset_path=args.new_data,
            device=device
        )
    
    else:
        parser.print_help()