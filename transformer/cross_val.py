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


def recover_cv_results(data_path, model_path_template, n_folds=5, device=None):
    """Recover cross-validation results from already trained models."""
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
            use_memory=True,
            progressive=True,
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
        fold_results = evaluate_model(
            model=fold_model,
            dataset=fold_test_dataset,
            build_dct_matrix=build_dct_matrix,
            build_idct_matrix=build_idct_matrix,
            batch_dct_transform=batch_dct_transform,
            batch_idct_transform=batch_idct_transform,
            device=device,
            batch_size=8,
            output_dir=f"../evaluations/fold_{fold_idx+1}"
        )
        
        # Store results
        all_fold_results.append(fold_results)
        
        # Check if this is the best fold
        avg_error = np.mean([v for v in fold_results['overall'].values()])
        if avg_error < best_fold_score:
            best_fold_score = avg_error
            best_fold_idx = fold_idx
    
    # Now aggregate the results properly
    if all_fold_results:
        # Compute average metrics across folds
        avg_results = {'overall': {}, 'by_table': {}, 'by_movement': {}}
        
        # Average overall metrics
        for ms in all_fold_results[0]['overall'].keys():
            values = [fold['overall'][ms] for fold in all_fold_results]
            avg_results['overall'][ms] = float(np.mean(values))
        
        # Create JSON data with NumPy conversion
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
        os.makedirs('../evaluations/cross_validation', exist_ok=True)
        with open('../evaluations/cross_validation/aggregated_results.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print("\n===== Cross-Validation Results =====")
        print("Average error across all folds:")
        for ms, value in avg_results['overall'].items():
            print(f"  {ms}ms: {value:.4f}")
        
        print(f"\nBest fold: {best_fold_idx+1} with score {best_fold_score:.4f}")
    else:
        print("No fold results collected.")
    
    return best_fold_idx, best_fold_score, model_path_template.replace('.pth', f'_fold_{best_fold_idx+1}.pth')


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