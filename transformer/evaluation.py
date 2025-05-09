"""
Hand Motion Prediction Model Evaluation Module.

This module contains functions for comprehensive evaluation of hand motion prediction models.
"""

import os
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import configuration
from config import (
    T_TOTAL, T_PRED, NUM_JOINTS, MASKED_JOINTS, JOINT_NAMES,
    EVAL_MS, MS_TO_FRAMES, EVALUATIONS_DIR
)
from evaluation_utils import (
    plot_per_joint_error,
    plot_error_progression_by_frame,
    plot_action_comparison,
    generate_detailed_error_table,
    generate_joint_group_table,
    analyze_dct_coefficients,
    visualize_predictions
)


def evaluate_model(model, dataset, 
                   build_dct_matrix, build_idct_matrix, 
                   batch_dct_transform, batch_idct_transform,
                   device=None, batch_size=32, output_dir=EVALUATIONS_DIR):
    """Comprehensive evaluation protocol for hand motion prediction models.
    
    Args:
        model: Trained motion prediction model
        dataset: Dataset containing test samples
        build_dct_matrix: Function to build DCT transformation matrix
        build_idct_matrix: Function to build inverse DCT transformation matrix
        batch_dct_transform: Function to apply DCT to a batch of sequences
        batch_idct_transform: Function to apply inverse DCT to a batch of sequences
        device: Device to run evaluation on (default: None, will use CUDA if available)
        batch_size: Batch size for evaluation (default: 32)
        output_dir: Directory to save evaluation outputs (default from config)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Precompute DCT and IDCT matrices
    dct_m = build_dct_matrix(T_TOTAL).to(device)
    idct_m = build_idct_matrix(T_PRED).to(device)
    
    # Get evaluation frames from config
    eval_frames = [MS_TO_FRAMES[ms] for ms in EVAL_MS]
    
    # Track metrics by exercise table and movement ID
    table_results = {}  # Results grouped by exercise table only (B or C)
    detailed_results = {}  # Results grouped by exercise table AND movement ID (e.g., B_1)
    
    # For additional analysis
    all_preds = []
    all_targets = []
    all_metadata = []
    sample_input_dct = None
    sample_baseline_dct = None
    sample_pred_residual = None
    
    # Switch to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (obs_batch, fut_batch, metadata) in enumerate(tqdm(loader, desc="Evaluating")):
            B = obs_batch.size(0)
            obs_batch = obs_batch.to(device)
            fut_batch = fut_batch.to(device)
            
            # Process each sample in the batch
            for i in range(B):
                # Get exercise table and movement ID for improved categorization
                exercise_table = str(metadata['exercise_table'][i])
                movement_id = str(metadata['movement_id'][i])
                
                # Create combined category for detailed analysis 
                detailed_category = f"{exercise_table}_{movement_id}"
                
                # Initialize dictionaries if needed
                if exercise_table not in table_results:
                    table_results[exercise_table] = {ms: [] for ms in EVAL_MS}
                
                if detailed_category not in detailed_results:
                    detailed_results[detailed_category] = {ms: [] for ms in EVAL_MS}
                
                # Standard prediction process
                obs = obs_batch[i].unsqueeze(0)
                fut = fut_batch[i].unsqueeze(0)
                
                # Prepare input with padding
                last_frame = obs[:, -1:, :]
                padded_obs = torch.cat([obs, last_frame.repeat(1, T_PRED, 1)], dim=1)
                
                # Apply batch DCT
                input_dct = batch_dct_transform(padded_obs, dct_m)
                
                # Forward pass
                predicted_residual = model(input_dct)
                
                # Extract future part
                pred_residual = predicted_residual[:, -T_PRED:, :]
                baseline_dct = input_dct[:, -T_PRED:, :]
                predicted_dct = pred_residual + baseline_dct
                
                # Convert back to time domain
                pred_fut = batch_idct_transform(predicted_dct, idct_m)
                
                # Apply mask for masked joints (thumb joints and LFJ5)
                mask = torch.ones(pred_fut.shape[-1], dtype=torch.bool, device=device)
                mask[MASKED_JOINTS] = False
                
                # Save sample for additional analysis (first batch)
                if batch_idx == 0 and i == 0:
                    sample_input_dct = input_dct.clone().cpu()
                    sample_baseline_dct = baseline_dct.clone().cpu()
                    sample_pred_residual = pred_residual.clone().cpu()
                
                # Store predictions and targets for additional analysis
                if len(all_preds) < 10:  # Limit to 10 samples for memory efficiency
                    all_preds.append(pred_fut.clone().cpu())
                    all_targets.append(fut.clone().cpu())
                    all_metadata.append({k: v[i] for k, v in metadata.items()})
                
                # Calculate error at each evaluation frame
                for f_idx, frame in enumerate(eval_frames):
                    if frame < fut.shape[1]:  # Make sure frame exists
                        # MAE for angle representation
                        error = (pred_fut[0, frame, mask] - fut[0, frame, mask]).abs().mean().item()
                        table_results[exercise_table][EVAL_MS[f_idx]].append(error)
                        detailed_results[detailed_category][EVAL_MS[f_idx]].append(error)
                
                # Visualization (limited to first few samples)
                try:
                    if batch_idx < 3 and i == 0:
                        visualize_predictions(pred_fut[0], fut[0], batch_idx, 
                                            f"{exercise_table} - Movement {movement_id}", NUM_JOINTS)
                except (NameError, ImportError):
                    pass  # Skip visualization if function not available
    
    # Calculate average metrics by exercise table
    overall_results = {ms: [] for ms in EVAL_MS}
    for table, metrics in table_results.items():
        for ms, errors in metrics.items():
            if errors:
                avg_error = sum(errors) / len(errors)
                overall_results[ms].append(avg_error)
    
    # Compute overall averages
    overall_avg = {ms: sum(errors)/len(errors) if errors else float('nan') for ms, errors in overall_results.items()}
    
    # Print results in format similar to papers - grouped by exercise table
    print("\n----- Evaluation Results (MAE) by Exercise Table -----")
    header = ["Table"] + [f"{ms}ms" for ms in EVAL_MS]
    print("  ".join(f"{h:<12}" for h in header))
    print("-" * 100)
    
    for table, metrics in table_results.items():
        row = [table[:12]]
        for ms in EVAL_MS:
            if metrics[ms]:
                row.append(f"{sum(metrics[ms])/len(metrics[ms]):.4f}")
            else:
                row.append("N/A")
        print("  ".join(f"{r:<12}" for r in row))
    
    print("-" * 100)
    row = ["Average"]
    for ms in EVAL_MS:
        if overall_results[ms]:
            row.append(f"{overall_avg[ms]:.4f}")
        else:
            row.append("N/A")
    print("  ".join(f"{r:<12}" for r in row))
    
    # Print detailed results - top 5 best and worst movements
    print("\n----- Top 5 Best/Worst Movements (at 400ms) -----")
    
    # Get all detailed categories with their average error at 400ms
    movement_errors = []
    for category, metrics in detailed_results.items():
        if metrics[400] and len(metrics[400]) >= 5:  # Only consider categories with enough samples
            avg_error = sum(metrics[400]) / len(metrics[400])
            movement_errors.append((category, avg_error))
    
    # Sort by error (ascending)
    movement_errors.sort(key=lambda x: x[1])
    
    # Display best 5
    print("Best performing movements:")
    for i, (category, error) in enumerate(movement_errors[:5]):
        print(f"{i+1}. {category}: {error:.4f}")
    
    # Display worst 5
    print("\nWorst performing movements:")
    for i, (category, error) in enumerate(movement_errors[-5:]):
        print(f"{i+1}. {category}: {error:.4f}")
    
    # Export results to CSV for paper tables
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Export table-based results
        with open(f'{output_dir}/evaluation_results_by_table.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Exercise Table'] + [f'{ms}ms' for ms in EVAL_MS])
            
            for table, metrics in table_results.items():
                row = [table]
                for ms in EVAL_MS:
                    if metrics[ms]:
                        row.append(sum(metrics[ms]) / len(metrics[ms]))
                    else:
                        row.append('N/A')
                writer.writerow(row)
            
            # Add average row
            avg_row = ['Average']
            for ms in EVAL_MS:
                if overall_results[ms]:
                    avg_row.append(sum(overall_results[ms]) / len(overall_results[ms]))
                else:
                    avg_row.append('N/A')
            writer.writerow(avg_row)
        
        # Export detailed results
        with open(f'{output_dir}/evaluation_results_detailed.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Table_Movement'] + [f'{ms}ms' for ms in EVAL_MS])
            
            for category, metrics in detailed_results.items():
                row = [category]
                for ms in EVAL_MS:
                    if metrics[ms]:
                        row.append(sum(metrics[ms]) / len(metrics[ms]))
                    else:
                        row.append('N/A')
                writer.writerow(row)
            
        print(f"Results exported to {output_dir}/evaluation_results_by_table.csv and {output_dir}/evaluation_results_detailed.csv")
    except Exception as e:
        print(f"Could not export results to CSV: {e}")
    
    # Additional analysis and visualizations
    print("\n----- Generating Additional Analysis -----")
    
    # Create directories for output
    plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Only proceed with additional analysis if we have data and evaluation utilities
    if all_preds and all_targets:
        try:
            # 1. Per-Joint Error Analysis
            try:
                # Stack first few predictions and targets
                stacked_preds = torch.cat(all_preds[:5], dim=0)
                stacked_targets = torch.cat(all_targets[:5], dim=0)
                # Create mask on CPU
                cpu_mask = torch.ones(stacked_preds.shape[-1], dtype=torch.bool)
                cpu_mask[MASKED_JOINTS] = False
                
                plot_per_joint_error(stacked_preds, stacked_targets, cpu_mask, JOINT_NAMES)
                print("✅ Generated per-joint error analysis")
            except (NameError, ImportError) as e:
                print(f"Could not generate per-joint error analysis: {e}")
                
            # 2. Error Progression Analysis
            try:
                plot_error_progression_by_frame(stacked_preds, stacked_targets, cpu_mask)
                print("✅ Generated error progression analysis")
            except (NameError, ImportError) as e:
                print(f"Could not generate error progression analysis: {e}")
            
            # 3. Action Comparison Visualization
            try:
                plot_action_comparison(table_results, EVAL_MS)
                print("✅ Generated action comparison visualization")
            except (NameError, ImportError) as e:
                print(f"Could not generate action comparison: {e}")
            
            # 4. Detailed Error Table
            try:
                # Generate for both table results and detailed results
                generate_detailed_error_table(table_results, EVAL_MS, output_path=f'{output_dir}/table_error_breakdown.csv')
                generate_detailed_error_table(detailed_results, EVAL_MS, output_path=f'{output_dir}/movement_error_breakdown.csv')
                print("✅ Generated detailed error tables")
            except (NameError, ImportError) as e:
                print(f"Could not generate detailed error tables: {e}")
            
            # 5. Joint Group Analysis
            try:
                generate_joint_group_table(stacked_preds, stacked_targets, JOINT_NAMES)
                print("✅ Generated joint group analysis")
            except (NameError, ImportError) as e:
                print(f"Could not generate joint group analysis: {e}")
            
            # 6. DCT Coefficient Analysis
            if sample_input_dct is not None and sample_baseline_dct is not None and sample_pred_residual is not None:
                try:
                    analyze_dct_coefficients(sample_input_dct, sample_baseline_dct, sample_pred_residual)
                    print("✅ Generated DCT coefficient analysis")
                except (NameError, ImportError) as e:
                    print(f"Could not generate DCT coefficient analysis: {e}")
        
        except Exception as e:
            print(f"Error during additional analysis: {e}")
    else:
        print("⚠️ No samples available for additional analysis")
    
    # Return the computed metrics
    return {
        "by_table": table_results,
        "by_movement": detailed_results,
        "overall": overall_avg
    }