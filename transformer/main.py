"""
Main script for training and evaluating hand motion prediction models.

This script handles command-line arguments and runs the training and evaluation workflow.
"""

import os
import json
import argparse
import pickle
import torch
import h5py

# Import configuration
from config import (
    T_OBS, T_PRED, DEFAULT_MODEL_PATH
)

# Import model and utilities
from model import create_model
from dataset import MovementDataset, create_simulator_dataset
from training_utils import (
    train_model,
    create_subject_split,
    create_movement_split,
    run_cross_validation
)
from evaluation import evaluate_model
from dct_utils import (
    build_dct_matrix,
    build_idct_matrix,
    batch_dct_transform,
    batch_idct_transform
)


def main():
    """Main function to train and test the model."""
    parser = argparse.ArgumentParser(description="Train and test the hand motion prediction model")
    parser.add_argument('--data', type=str, default="../datasets/shadow_dataset.h5", help="Path to the dataset")
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both', help="Operation mode")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--subset-size', type=int, default=None, help="Use a subset of data for debugging")
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help="Path for saving/loading model")
    
    # Dataset filtering options
    parser.add_argument('--save-dataset', action='store_true', help="Save filtered HDF5 movement keys")
    parser.add_argument('--load-dataset', action='store_true', help="Load filtered HDF5 movement keys")
    parser.add_argument('--filtered-keys', type=str, default="filtered_keys.pkl", help="Path for saving/loading filtered keys")
    parser.add_argument('--filter-save', action='store_true', help="Filter and save a clean dataset")
    parser.add_argument('--filtered-output', type=str, default="../datasets/shadow_dataset_filtered.h5", 
                   help="Output path for filtered dataset")
    # Data splitting parameters
    parser.add_argument('--split-type', choices=['subject', 'movement', 'random'], default='subject',
                      help="How to split the dataset for train/test (by subject, movement, or random)")
    parser.add_argument('--test-ratio', type=float, default=0.2, 
                      help="Percentage of data to use for testing (for random split)")
    parser.add_argument('--test-subjects', type=str, nargs='+', 
                      help="Specific subjects to use for testing (for subject split)")
    parser.add_argument('--cross-val', action='store_true', help="Use cross-validation")
    parser.add_argument('--n-folds', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--create-sim-data', action='store_true', 
                   help="Create a dataset of predicted movements for simulator visualization")
    parser.add_argument('--sim-data-path', type=str, default="../sim_data/predictions.h5", 
                    help="Path to save simulator-compatible dataset")
    parser.add_argument('--sim-samples', type=int, default=100, 
                    help="Number of samples to include in simulator dataset")

    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if we should load prefiltered keys
    filtered_keys = None
    if args.load_dataset and os.path.exists(args.filtered_keys):
        print(f"📂 Loading prefiltered keys from {args.filtered_keys}")
        with open(args.filtered_keys, 'rb') as f:
            filtered_keys = pickle.load(f)
        print(f"Found {len(filtered_keys)} prefiltered keys")
    
    # Create dataset
    dataset = MovementDataset(
        h5_path=args.data,
        T_obs=T_OBS,
        T_pred=T_PRED,
        subset_size=args.subset_size,
        prefiltered=(args.load_dataset and filtered_keys is not None),
        filtered_keys=filtered_keys
    )
    
    # Save filtered keys if requested
    if args.save_dataset:
        print(f"💾 Saving filtered keys to {args.filtered_keys}")
        os.makedirs(os.path.dirname(args.filtered_keys), exist_ok=True)
        with open(args.filtered_keys, 'wb') as f:
            pickle.dump(dataset.valid_samples, f)
        print(f"✅ Saved {len(dataset.valid_samples)} filtered keys")
    
    # Filter and save dataset if requested  
    if args.filter_save:
        print(f"💾 Saving filtered dataset to {args.filtered_output} ...")
        os.makedirs(os.path.dirname(args.filtered_output), exist_ok=True)
        with h5py.File(args.data, 'r') as h5_in, h5py.File(args.filtered_output, 'w') as h5_out:
            in_grp = h5_in['movements']
            out_grp = h5_out.create_group('movements')
            for key in dataset.valid_samples:
                in_grp.copy(key, out_grp)
        print(f"✅ Filtered dataset saved to {args.filtered_output}")
    
    # Create data splits for train/test
    train_keys, test_keys = None, None
    
    if args.cross_val:
        best_model, train_keys, test_keys, test_dataset_for_sim = run_cross_validation(dataset, args, device)
    else:
        # Create a single train/test split
        if args.split_type == 'subject':
            train_keys, test_keys = create_subject_split(dataset, test_subjects=args.test_subjects, test_ratio=args.test_ratio)
        elif args.split_type == 'movement':
            train_keys, test_keys = create_movement_split(dataset, test_ratio=args.test_ratio)
        else:  # random split
            # Implement a simple random split
            import random
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            split = int(len(indices) * (1 - args.test_ratio))
            train_indices = indices[:split]
            test_indices = indices[split:]
            train_keys = [dataset.valid_samples[i] for i in train_indices]
            test_keys = [dataset.valid_samples[i] for i in test_indices]
            print(f"Random split: {len(train_keys)} training samples, {len(test_keys)} testing samples")

    # Create train and test datasets
    train_dataset = MovementDataset(
        h5_path=args.data,
        T_obs=T_OBS,
        T_pred=T_PRED,
        prefiltered=True,
        filtered_keys=train_keys,
        dataset_type="training"
    )
    
    test_dataset = MovementDataset(
        h5_path=args.data,
        T_obs=T_OBS,
        T_pred=T_PRED,
        prefiltered=True,
        filtered_keys=test_keys,
        dataset_type="test"
    )

    
    if not args.cross_val:
        # Train model
        if args.mode in ['train', 'both']:
            # Create model
            model = create_model(
                d_model=128,
                nhead=8,
                num_layers=3,
                dim_feedforward=512,
                device=device
            )
            
            # Train model
            model = train_model(
                model=model,
                dataset=train_dataset,  # Use training dataset for training
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                save_path=args.model_path,
                device=device
            )
        
        # Test model using the modular evaluation function
        if args.mode in ['test', 'both']:
            if args.mode == 'test':
                # Create model and load weights
                model = create_model(
                    d_model=128,
                    nhead=8,
                    num_layers=3,
                    dim_feedforward=512,
                    device=device
                )
                model.load_state_dict(torch.load(args.model_path, map_location=device))
            
            # Use the modular evaluation function
            results = evaluate_model(
                model=model, 
                dataset=test_dataset,
                build_dct_matrix=build_dct_matrix,
                build_idct_matrix=build_idct_matrix,
                batch_dct_transform=batch_dct_transform,
                batch_idct_transform=batch_idct_transform,
                device=device, 
                batch_size=args.batch_size
            )
            
            # Save results
            os.makedirs(os.path.dirname("../evaluations/evaluation_results.json"), exist_ok=True)
            with open("../evaluations/evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Saved ../evaluations/evaluation_results.json")


        if args.create_sim_data:
            # Load model if needed
            if 'model' not in locals():
                model = create_model(
                    d_model=128,
                    nhead=8,
                    num_layers=3,
                    dim_feedforward=512,
                    use_memory=not args.no_memory,
                    progressive=not args.no_progressive,
                    device=device
                )
                model.load_state_dict(torch.load(args.model_path, map_location=device))
            
            # Create simulator dataset
            create_simulator_dataset(
                model=model,
                dataset=test_dataset,
                output_path=args.sim_data_path,
                num_samples=args.sim_samples,
                device=device
            )
        
        

if __name__ == "__main__":
    main()