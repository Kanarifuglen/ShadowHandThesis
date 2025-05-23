"""
Training utility functions for hand motion prediction models.

This module contains functions for dataset splitting, model training,
and other training-related utilities.
"""

import os
import pickle
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import MovementDataset, create_simulator_dataset
from model import create_model
from evaluation import evaluate_model
import json
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import configuration
from config import (
    T_OBS, T_PRED, T_TOTAL, NUM_JOINTS, MASKED_JOINTS,
    PLOTS_DIR, DEFAULT_MODEL_PATH
)

# Import DCT transformation utilities
from dct_utils import (
    build_dct_matrix, build_idct_matrix,
    batch_dct_transform, batch_idct_transform
)


# === Data Splitting Functions for Proper Evaluation ===
def create_subject_split(dataset, test_subjects=None, test_ratio=0.2):
    """Split dataset by subject_id"""
    # Get unique subjects
    unique_subjects = set()
    for key in dataset.valid_samples:
        subject = dataset.h5['movements'][key].attrs.get('subject_id', "Unknown")
        unique_subjects.add(subject)
    unique_subjects = list(unique_subjects)
    
    # If no test subjects provided, randomly select some
    if test_subjects is None:
        num_test_subjects = max(1, int(len(unique_subjects) * test_ratio))
        random.shuffle(unique_subjects)
        test_subjects = unique_subjects[:num_test_subjects]
    
    # Create the split
    train_keys = []
    test_keys = []
    
    for key in dataset.valid_samples:
        subject = dataset.h5['movements'][key].attrs.get('subject_id', "Unknown")
        if subject in test_subjects:
            test_keys.append(key)
        else:
            train_keys.append(key)
    
    print(f"Subject split: {len(train_keys)} training samples, {len(test_keys)} testing samples")
    print(f"Test subjects: {test_subjects}")
    return train_keys, test_keys


def create_movement_split(dataset, test_ratio=0.2):
    """Split dataset by movement_name/exercise_id"""
    # Get unique movement types
    movement_types = set()
    for key in dataset.valid_samples:
        movement = dataset.h5['movements'][key].attrs.get('exercise_id', "Unknown")
        movement_types.add(movement)
    
    # Randomly select test movements
    movement_types = list(movement_types)
    num_test_movements = max(1, int(len(movement_types) * test_ratio))
    random.shuffle(movement_types)
    test_movements = movement_types[:num_test_movements]
    
    # Create split
    train_keys = []
    test_keys = []
    for key in dataset.valid_samples:
        movement = dataset.h5['movements'][key].attrs.get('exercise_id', "Unknown")
        if movement in test_movements:
            test_keys.append(key)
        else:
            train_keys.append(key)
    
    print(f"Movement split: {len(train_keys)} training samples, {len(test_keys)} testing samples")
    print(f"Test movements: {test_movements}")
    return train_keys, test_keys


def create_cross_validation_splits(dataset, n_folds=5, by='subject_id'):
    """Create n-fold cross-validation splits by subject, movement, or session"""
    # Get unique values for the split criterion
    unique_values = set()
    for key in dataset.valid_samples:
        value = dataset.h5['movements'][key].attrs.get(by, "Unknown")
        unique_values.add(value)
    
    # Create folds
    unique_values = list(unique_values)
    random.shuffle(unique_values)
    fold_size = len(unique_values) // n_folds
    
    all_splits = []
    for i in range(n_folds):
        # Select test values for this fold
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else len(unique_values)
        test_values = unique_values[start_idx:end_idx]
        
        # Create train/test keys
        train_keys = []
        test_keys = []
        for key in dataset.valid_samples:
            value = dataset.h5['movements'][key].attrs.get(by, "Unknown")
            if value in test_values:
                test_keys.append(key)
            else:
                train_keys.append(key)
        
        all_splits.append((train_keys, test_keys))
    
    return all_splits


def train_model(model, dataset, epochs=20, batch_size=8, learning_rate=1e-4, 
                weight_decay=1e-5, save_path=DEFAULT_MODEL_PATH, device=None,
                use_weighted_sampling=False):
    """Train the motion prediction model.
    
    Args:
        model: The model to train
        dataset: Dataset for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        save_path: Path to save the best model
        device: Device to run training on
        use_weighted_sampling: Whether to use weighted sampling (for combined datasets)
        
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create DataLoader with appropriate sampler
    if use_weighted_sampling and hasattr(dataset, 'get_sample_weights'):
        # Get sample weights from combined dataset
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True
        )
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=sampler,  # Use weighted sampler
            num_workers=2,
            pin_memory=True
        )
        print("Using weighted sampling for combined dataset")
    else:
        # Standard DataLoader with shuffling
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5, 
        patience=5, 
        verbose=False
    )
    
    # Loss function
    l1_loss = nn.L1Loss() 
    
    # IDCT matrix for evaluation
    idct_mat = build_idct_matrix(T_PRED).to(device)
    
    # Training loop
    best_loss = float('inf')
    losses, maes = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_mae = 0, 0
        
        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as progress_bar:
            for obs_batch, fut_batch, metadata in progress_bar:
                B = obs_batch.size(0)
                obs_batch = obs_batch.to(device)  # (B, T_OBS, J)
                fut_batch = fut_batch.to(device)  # (B, T_PRED, J)
                
                # Prepare input sequence with padding for future frames
                last_frame = obs_batch[:, -1:, :]  # (B, 1, J)
                padded_obs = torch.cat([
                    obs_batch, 
                    last_frame.repeat(1, T_PRED, 1)
                ], dim=1)  # (B, T_TOTAL, J)
                
                # Use batch DCT transform for more efficient processing
                dct_m = build_dct_matrix(T_TOTAL).to(device)
                input_dct = batch_dct_transform(padded_obs, dct_m)
                
                # Extract baseline DCT (last T_PRED coefficients)
                baseline_dct = input_dct[:, -T_PRED:, :].clone()  # (B, T_PRED, J)
                
                # Forward pass
                optimizer.zero_grad()
                predicted_residual = model(input_dct)  # (B, T_TOTAL, J)
                
                # Extract prediction residuals for future frames - use clone to avoid in-place issues
                pred_residual = predicted_residual[:, -T_PRED:, :].clone()  # (B, T_PRED, J)
                
                # Add residual to baseline - use addition instead of in-place add
                predicted_dct = pred_residual + baseline_dct  # (B, T_PRED, J)
                
                # Apply IDCT to get time domain prediction
                pred_fut = batch_idct_transform(predicted_dct, idct_mat)  # (B, T_PRED, J)
                
                # Create mask for masked joints (thumb and LFJ5)
                mask = torch.ones(pred_fut.shape[-1], dtype=torch.bool, device=device)
                mask[MASKED_JOINTS] = False
                
                # Apply mask
                masked_pred = pred_fut[:, :, mask]
                masked_target = fut_batch[:, :, mask]
                
                # Compute loss
                loss = l1_loss(masked_pred, masked_target)
                
                # Backpropagation
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Compute Mean Absolute Error
                with torch.no_grad():
                    mae = (masked_pred - masked_target).abs().mean().item()
                
                # Update statistics
                total_loss += loss.item()
                total_mae += mae
                
                # Add dataset source to progress bar if available
                postfix = {
                    'loss': loss.item(),
                    'mae': f"{mae:.4f}"
                }
                
                # If using a combined dataset, track dataset distribution
                if 'source_dataset' in metadata:
                    sources = metadata['source_dataset']
                    sources_count = {}
                    for source in sources:
                        if source in sources_count:
                            sources_count[source] += 1
                        else:
                            sources_count[source] = 1
                    
                    source_info = "/".join([f"{k}:{v}" for k, v in sources_count.items()])
                    postfix['sources'] = source_info
                
                # Update progress bar
                progress_bar.set_postfix(postfix)
        
        # Calculate average metrics
        avg_loss = total_loss / len(loader)
        avg_mae = total_mae / len(loader)
        
        losses.append(avg_loss)
        maes.append(avg_mae)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, MAE={avg_mae:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Make sure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model saved to {save_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss (L1)")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(maes)
    plt.title("Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (radians)")
    
    plt.tight_layout()
    
    # Make sure plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/training_curves.png")
    print(f"📊 Saved training curves to '{PLOTS_DIR}/training_curves.png'")

    return model


def run_cross_validation(dataset, args, device, skip_training=False):
    """
    Run full k-fold cross-validation with enhanced thesis-specific evaluations.
    
    Args:
        dataset: The full dataset to split into folds
        args: Command line arguments
        device: Device to run on (CPU/GPU)
    
    Returns:
        tuple: (best_model, train_keys, test_keys, best_test_dataset)
    """
    print(f"Creating {args.n_folds}-fold cross-validation splits by {args.split_type}...")
    cv_splits = create_cross_validation_splits(dataset, n_folds=args.n_folds, by=args.split_type + '_id')
    
    # Save the CV splits for reproducibility
    os.makedirs('../evaluations/cross_validation', exist_ok=True)
    with open('../evaluations/cross_validation/cv_splits.pkl', 'wb') as f:
        pickle.dump(cv_splits, f)
    print(f"Saved cross-validation splits to '../evaluations/cross_validation/cv_splits.pkl'")
    
    # Store results across folds
    all_fold_results = []
    best_fold_score = float('inf')
    best_fold_model = None
    best_fold_idx = 0
    best_fold_test_dataset = None
    
    # Define target horizons to match thesis requirements
    target_horizons = [80, 160, 320, 400, 600, 1000]  # in ms
    
    # Loop through all folds
    for fold_idx, (fold_train_keys, fold_test_keys) in enumerate(cv_splits):
        print(f"\n===== Processing Fold {fold_idx+1}/{args.n_folds} =====")
        print(f"Training set: {len(fold_train_keys)} samples, Test set: {len(fold_test_keys)} samples")
        
        # Create datasets for this fold
        fold_train_dataset = MovementDataset(
            h5_path=args.data,
            T_obs=T_OBS,
            T_pred=T_PRED,
            prefiltered=True,
            filtered_keys=fold_train_keys,
            dataset_type=f"train_fold_{fold_idx+1}"
        )
        
        fold_test_dataset = MovementDataset(
            h5_path=args.data,
            T_obs=T_OBS,
            T_pred=T_PRED,
            prefiltered=True,
            filtered_keys=fold_test_keys,
            dataset_type=f"test_fold_{fold_idx+1}"
        )
        
        # Set fold-specific model path
        fold_model_path = args.model_path.replace('.pth', f'_fold_{fold_idx+1}.pth')
        
        # Create model for this fold
        fold_model = create_model(
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            device=device
        )
        
        # If skipping training, try to load existing model
        if skip_training:
            if os.path.exists(fold_model_path):
                print(f"Loading existing model from {fold_model_path}")
                fold_model.load_state_dict(torch.load(fold_model_path, map_location=device))
            else:
                print(f"WARNING: Skip-training specified but no model found at {fold_model_path}")
        # Otherwise train the model if needed
        elif args.mode in ['train', 'both']:
            # Train model for this fold
            fold_model = train_model(
                model=fold_model,
                dataset=fold_train_dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                save_path=fold_model_path,
                device=device
            )
        else:  # Only testing
            if os.path.exists(fold_model_path):
                fold_model.load_state_dict(torch.load(fold_model_path, map_location=device))
            else:
                print(f"Warning: No model found at {fold_model_path}, using default path")
                fold_model.load_state_dict(torch.load(args.model_path, map_location=device))
            
            # Evaluate model on this fold
            if args.mode in ['test', 'both']:
                # Create fold-specific output directory
                fold_output_dir = f"../evaluations/fold_{fold_idx+1}"
                os.makedirs(fold_output_dir, exist_ok=True)
                
                # Run enhanced evaluation
                fold_results = evaluate_model(
                    model=fold_model,
                    dataset=fold_test_dataset,
                    build_dct_matrix=build_dct_matrix,
                    build_idct_matrix=build_idct_matrix,
                    batch_dct_transform=batch_dct_transform,
                    batch_idct_transform=batch_idct_transform,
                    device=device,
                    batch_size=args.batch_size,
                    output_dir=fold_output_dir
                )
                
                # Store results for this fold
                all_fold_results.append(fold_results)
                
                # Track best fold based on average error (lower is better)
                avg_error = np.mean([v for v in fold_results['overall'].values()])
                if avg_error < best_fold_score:
                    best_fold_score = avg_error
                    best_fold_model = fold_model
                    best_fold_idx = fold_idx
                    best_fold_test_dataset = fold_test_dataset
        
    # Generate thesis-specific analyses using recover_cv_results
    if args.mode in ['test', 'both'] and all_fold_results:
        # First aggregate results as before for backward compatibility
        aggregate_cv_results(all_fold_results, best_fold_score, best_fold_idx)
        
        # Now use the enhanced recover_cv_results to generate all thesis visualizations
        from cross_val import recover_cv_results
        
        print("\n===== Generating Thesis-Specific Visualizations and Analyses =====")
        best_idx, best_score, best_path = recover_cv_results(
            args.data, 
            args.model_path, 
            args.n_folds, 
            device
        )
        
        # If the best model from recover_cv_results differs, print a warning
        if best_idx != best_fold_idx:
            print(f"Warning: Best fold from recover_cv_results ({best_idx+1}) differs from training ({best_fold_idx+1})")
            print(f"Using fold {best_fold_idx+1} for return values as it was selected during training")
        
        # Ensure the thesis outputs are created for the best fold
        best_fold_model_path = args.model_path.replace('.pth', f'_fold_{best_fold_idx+1}.pth')
        thesis_output_dir = f"../evaluations/thesis_outputs"
        os.makedirs(thesis_output_dir, exist_ok=True)
        
        # Create a symbolic link to the best fold's evaluation outputs
        if os.path.exists(f"../evaluations/fold_{best_fold_idx+1}"):
            for file in os.listdir(f"../evaluations/fold_{best_fold_idx+1}"):
                if os.path.isfile(f"../evaluations/fold_{best_fold_idx+1}/{file}"):
                    # Copy or link best fold's files to thesis outputs
                    shutil.copy2(
                        f"../evaluations/fold_{best_fold_idx+1}/{file}",
                        f"{thesis_output_dir}/{file}"
                    )
        
        # Save best model from cross-validation
        best_model_path = args.model_path.replace('.pth', '_best_cv.pth')
        torch.save(best_fold_model.state_dict(), best_model_path)
        print(f"\n💾 Best cross-validation model saved to {best_model_path}")
        
        # Create simulator data using the best model and its test dataset
        if args.create_sim_data:
            sim_path = args.sim_data_path.replace('.h5', '_cv_best.h5')
            create_simulator_dataset(
                model=best_fold_model,
                dataset=best_fold_test_dataset,
                output_path=sim_path,
                num_samples=args.sim_samples,
                device=device
            )
    
    # Return best model, first fold splits, and best test dataset
    return best_fold_model, cv_splits[0][0], cv_splits[0][1], best_fold_test_dataset


def aggregate_cv_results(all_fold_results, best_fold_score, best_fold_idx):
    """Helper function to aggregate cross-validation results."""
    # Compute average metrics across folds
    avg_results = {'overall': {}, 'by_table': {}, 'by_movement': {}}
    
    # Average overall metrics
    for ms in all_fold_results[0]['overall'].keys():
        values = [fold['overall'][ms] for fold in all_fold_results if ms in fold['overall']]
        if values:  # Only compute mean if values exist
            avg_results['overall'][ms] = float(np.mean(values))
    
    # Save aggregated results
    os.makedirs('../evaluations/cross_validation', exist_ok=True)
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

    # Now it's safe to dump to JSON
    with open('../evaluations/cross_validation/aggregated_results.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Print aggregated results
    print("\n===== Cross-Validation Results =====")
    print("Average error across all folds:")
    for ms, value in avg_results['overall'].items():
        print(f"  {ms}ms: {value:.4f}")

def convert_numpy_to_python(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    import numpy as np
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