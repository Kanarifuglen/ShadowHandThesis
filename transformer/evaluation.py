"""
Hand Motion Prediction Model Evaluation Module.

This module contains functions for comprehensive evaluation of hand motion prediction models.
"""

import os

from matplotlib import pyplot as plt
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
                   device=None, batch_size=32, output_dir=None):
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
        output_dir: Directory to save evaluation outputs
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_dir is None:
        output_dir = EVALUATIONS_DIR
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Precompute DCT and IDCT matrices
    dct_m = build_dct_matrix(T_TOTAL).to(device)
    idct_m = build_idct_matrix(T_PRED).to(device)
    
    # Define time horizons to evaluate (in ms)
    eval_ms = [80, 160, 320, 400, 600, 1000]
    
    # Convert ms to frame indices (assuming 25fps = 40ms per frame)
    eval_frames = [int(ms / 40) for ms in eval_ms]
    
    # Track metrics by exercise table and movement ID
    table_results = {}  # Results grouped by exercise table
    movement_results = {}  # Results grouped by detailed movement
    joint_results = {}  # Results for each joint
    
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
                exercise_table = str(tensor_to_python(metadata['exercise_table'][i]))
                movement_id = str(tensor_to_python(metadata['movement_id'][i]))
                movement_name = str(tensor_to_python(metadata['movement_name'][i]))
                
                # Create combined category for detailed analysis 
                detailed_movement = f"{str(exercise_table)}_{str(movement_id)}: {str(movement_name)}"
                
                # Initialize dictionaries if needed
                if exercise_table not in table_results:
                    table_results[exercise_table] = {str(ms): [] for ms in eval_ms}
                
                if detailed_movement not in movement_results:
                    movement_results[detailed_movement] = {str(ms): [] for ms in eval_ms}
                
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
                        ms = eval_ms[f_idx]
                        ms_str = str(ms)
                        
                        # MAE for angle representation - average over masked joints
                        error = (pred_fut[0, frame, mask] - fut[0, frame, mask]).abs().mean().item()
                        
                        # Add to table and movement results
                        table_results[exercise_table][ms_str].append(tensor_to_python(error))
                        movement_results[detailed_movement][ms_str].append(tensor_to_python(error))

                        # Per-joint analysis at 400ms
                        if ms == 400 and frame < fut.shape[1]:
                            for j in range(pred_fut.shape[-1]):
                                if j not in MASKED_JOINTS:
                                    joint_name = JOINT_NAMES[j]
                                    joint_error = (pred_fut[0, frame, j] - fut[0, frame, j]).abs().item()
                                    
                                    if joint_name not in joint_results:
                                        joint_results[joint_name] = []
                                    
                                    joint_results[joint_name].append(tensor_to_python(joint_error))
                
                # Trajectory visualization (limited to first few samples)
                if batch_idx < 3 and i == 0:
                    try:
                        visualize_trajectories(pred_fut[0].cpu(), fut[0].cpu(), 
                                              f"{os.path.join(plots_dir, f'trajectory_batch{batch_idx}.png')}",
                                              f"{exercise_table} - {movement_name}")
                    except Exception as e:
                        print(f"Could not visualize trajectory: {e}")
    
    # Calculate average metrics by exercise table
    overall_results = {str(ms): [] for ms in eval_ms}
    
    for table, metrics in table_results.items():
        for ms_str, errors in metrics.items():
            if errors:
                avg_error = sum(errors) / len(errors)
                overall_results[ms_str].append(avg_error)
    
    # Compute overall averages
    overall_avg = {ms_str: sum(errors)/len(errors) if errors else float('nan') 
                 for ms_str, errors in overall_results.items()}
    
    # Print results in format similar to papers - grouped by exercise table
    print("\n----- Evaluation Results (MAE) by Exercise Table -----")
    header = ["Table"] + [f"{ms}ms" for ms in eval_ms]
    print("  ".join(f"{h:<12}" for h in header))
    print("-" * 100)
    
    for table, metrics in table_results.items():
        row = [table[:12]]
        for ms in eval_ms:
            ms_str = str(ms)
            if ms_str in metrics and metrics[ms_str]:
                row.append(f"{sum(metrics[ms_str])/len(metrics[ms_str]):.4f}")
            else:
                row.append("N/A")
        print("  ".join(f"{r:<12}" for r in row))
    
    print("-" * 100)
    row = ["Average"]
    for ms in eval_ms:
        ms_str = str(ms)
        if ms_str in overall_avg and not np.isnan(overall_avg[ms_str]):
            row.append(f"{overall_avg[ms_str]:.4f}")
        else:
            row.append("N/A")
    print("  ".join(f"{r:<12}" for r in row))
    
    # Print detailed results - top 5 best and worst movements
    print("\n----- Top 5 Best/Worst Movements (at 400ms) -----")
    
    # Get all detailed categories with their average error at 400ms
    movement_error_400ms = []
    for movement, metrics in movement_results.items():
        if "400" in metrics and metrics["400"] and len(metrics["400"]) >= 3:  # Only consider categories with enough samples
            avg_error = sum(metrics["400"]) / len(metrics["400"])
            movement_error_400ms.append((movement, avg_error))
    
    # Sort by error (ascending)
    movement_error_400ms.sort(key=lambda x: x[1])
    
    # Display best 5
    print("Best performing movements:")
    for i, (movement, error) in enumerate(movement_error_400ms[:5]):
        print(f"{i+1}. {movement}: {error:.4f}")
    
    # Display worst 5
    print("\nWorst performing movements:")
    for i, (movement, error) in enumerate(movement_error_400ms[-5:]):
        print(f"{i+1}. {movement}: {error:.4f}")
    
    # Generate thesis-specific outputs
    
    # 1. Overall performance table (Table ref:overall_performance)
    with open(f'{output_dir}/overall_performance_table.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time Horizon', 'Mean Absolute Error (radians)'])
        
        for ms in eval_ms:
            ms_str = str(ms)
            if ms_str in overall_avg and not np.isnan(overall_avg[ms_str]):
                writer.writerow([f"{ms}ms", f"{overall_avg[ms_str]:.4f}"])
    
    # 2. Error progression figure (Fig ref:error_progression)
    plt.figure(figsize=(10, 6))
    
    x_values = []
    y_values = []
    
    for ms in eval_ms:
        ms_str = str(ms)
        if ms_str in overall_avg and not np.isnan(overall_avg[ms_str]):
            x_values.append(ms)
            y_values.append(overall_avg[ms_str])
    
    plt.plot(x_values, y_values, 'o-', linewidth=2)
    plt.xlabel('Prediction Horizon (ms)')
    plt.ylabel('Mean Absolute Error (radians)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Error Progression Over Prediction Time Horizon')
    plt.savefig(f'{plots_dir}/error_progression.png', dpi=300)
    plt.close()
    
    # 3. Movement comparison figure (Fig ref:movement_comparison)
    plt.figure(figsize=(14, 8))
    
    # Get top and bottom movements
    selected_movements = [m for m, _ in movement_error_400ms[:5] + movement_error_400ms[-5:]]
    
    for movement in selected_movements:
        x = []
        y = []
        
        for ms in eval_ms:
            ms_str = str(ms)
            if ms_str in movement_results[movement] and movement_results[movement][ms_str]:
                avg_error = sum(movement_results[movement][ms_str]) / len(movement_results[movement][ms_str])
                x.append(ms)
                y.append(avg_error)
        
        # Truncate movement name for better display
        display_name = movement[:25] + "..." if len(movement) > 25 else movement
        plt.plot(x, y, 'o-', label=display_name)
    
    plt.xlabel('Prediction Horizon (ms)')
    plt.ylabel('Mean Absolute Error (radians)')
    plt.title('Performance Comparison Across Movement Types')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/movement_comparison.png', dpi=300)
    plt.close()
    
    # 4. Joint error analysis (Fig ref:joint_error)
    if joint_results:
        # Calculate average error per joint
        joint_avg_errors = {joint: sum(errors)/len(errors) if errors else 0 
                          for joint, errors in joint_results.items()}
        
        # Sort joints by name to maintain anatomical order
        sorted_joints = sorted(joint_avg_errors.keys(), key=lambda j: JOINT_NAMES.index(j) if j in JOINT_NAMES else 999)
        
        # Define joint groups
        joint_groups = {
            'Wrist': ['WRJ1', 'WRJ2'],
            'Thumb': ['THJ1', 'THJ2', 'THJ3', 'THJ4', 'THJ5'],
            'Index': ['FFJ1', 'FFJ2', 'FFJ3', 'FFJ4'],
            'Middle': ['MFJ1', 'MFJ2', 'MFJ3', 'MFJ4'],
            'Ring': ['RFJ1', 'RFJ2', 'RFJ3', 'RFJ4'],
            'Little': ['LFJ1', 'LFJ2', 'LFJ3', 'LFJ4', 'LFJ5']
        }
        
        # Assign colors to joint groups
        group_colors = {
            'Wrist': 'blue',
            'Thumb': 'green',
            'Index': 'red',
            'Middle': 'orange',
            'Ring': 'purple',
            'Little': 'brown'
        }
        
        # Determine color for each joint
        colors = []
        for joint in sorted_joints:
            color_assigned = False
            for group, joints in joint_groups.items():
                if joint in joints:
                    colors.append(group_colors[group])
                    color_assigned = True
                    break
            if not color_assigned:
                colors.append('gray')
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.bar(sorted_joints, [joint_avg_errors[j] for j in sorted_joints], color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.title("Per-Joint Prediction Error")
        plt.ylabel("Mean Absolute Error (radians)")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add legend for joint groups
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in group_colors.values()]
        plt.legend(handles, group_colors.keys(), loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/per_joint_error.png', dpi=300)
        plt.close()
        
        # 5. Joint group analysis (Table ref:joint_group)
        joint_group_errors = {group: [] for group in joint_groups}
        
        for joint, errors in joint_results.items():
            for group, joints in joint_groups.items():
                if joint in joints and errors:
                    joint_group_errors[group].extend(errors)
        
        # Generate joint group table
        with open(f'{output_dir}/joint_group_analysis.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Joint Group', 'Mean Error', 'Max Error', 'Min Error', 'Std Deviation'])
            
            for group, errors in joint_group_errors.items():
                if errors:
                    writer.writerow([
                        group,
                        f"{np.mean(errors):.4f}",
                        f"{np.max(errors):.4f}",
                        f"{np.min(errors):.4f}",
                        f"{np.std(errors):.4f}"
                    ])
    
    # 6. DCT coefficient analysis (Table ref:dct_coefficients)
    if sample_input_dct is not None and sample_baseline_dct is not None and sample_pred_residual is not None:
        # Number of coefficients to analyze
        num_coeffs = min(20, sample_input_dct.shape[1])
        
        # Calculate average magnitude across batch and joints
        input_mag = sample_input_dct[:, :num_coeffs, :].abs().mean(dim=(0, 2)).numpy()
        baseline_mag = sample_baseline_dct[:, :num_coeffs, :].abs().mean(dim=(0, 2)).numpy()
        residual_mag = sample_pred_residual[:, :num_coeffs, :].abs().mean(dim=(0, 2)).numpy()
        
        # Save in CSV format for paper table
        with open(f'{output_dir}/dct_coefficient_analysis.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Coefficient Index', 'Input Magnitude', 'Baseline Magnitude', 'Residual Magnitude'])
            
            for i in range(num_coeffs):
                writer.writerow([
                    i, 
                    f"{input_mag[i]:.6f}", 
                    f"{baseline_mag[i]:.6f}", 
                    f"{residual_mag[i]:.6f}"
                ])
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        x = np.arange(num_coeffs)
        plt.plot(x, input_mag, 'o-', label='Input DCT Coefficients', linewidth=2)
        plt.plot(x, baseline_mag, 's--', label='Baseline DCT Coefficients', linewidth=2)
        plt.plot(x, residual_mag, '^-', label='Predicted Residual Coefficients', linewidth=2)
        
        plt.xlabel('DCT Coefficient Index')
        plt.ylabel('Average Magnitude')
        plt.title('DCT Coefficient Magnitude Analysis')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/dct_coefficient_analysis.png', dpi=300)
        plt.close()
    
    # Create qualitative trajectory visualizations
    if all_preds and all_targets:
        try:
            # Combined qualitative analysis
            visualize_qualitative_comparison(all_preds[:3], all_targets[:3], 
                                            [m.get('movement_name', 'Unknown') for m in all_metadata[:3]],
                                            f"{plots_dir}/qualitative_comparison.png")
        except Exception as e:
            print(f"Could not generate qualitative comparison: {e}")
    
    # Return the computed metrics in a structured format
    return tensor_to_python ({
        "by_table": table_results,
        "by_movement": movement_results,
        "by_joint": joint_results,
        "overall": overall_avg
    })


def visualize_trajectories(pred, target, output_path, title="Joint Trajectories"):
    """Visualize trajectories of key joints over time.
    
    Args:
        pred: Predicted poses tensor
        target: Ground truth poses tensor
        output_path: Path to save the figure
        title: Title for the figure
    """
    plt.figure(figsize=(15, 10))
    
    # Select representative joints
    key_joints = [0, 6, 10, 14, 18, 23]  # Wrist, Thumb tip, Index tip, Middle tip, Ring tip, Little tip
    joint_labels = ["Wrist (WRJ1)", "Thumb tip (THJ5)", "Index tip (FFJ4)", 
                   "Middle tip (MFJ4)", "Ring tip (RFJ4)", "Little tip (LFJ5)"]
    
    for j_idx, joint_idx in enumerate(key_joints):
        plt.subplot(len(key_joints), 1, j_idx + 1)
        plt.plot(target[:, joint_idx].cpu().numpy(), 'g-', label='Ground Truth')
        plt.plot(pred[:, joint_idx].cpu().numpy(), 'r--', label='Prediction')
        plt.title(f"{joint_labels[j_idx]} Trajectory")
        if j_idx == 0:
            plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def visualize_qualitative_comparison(preds, targets, movement_names, output_path):
    """Create a side-by-side comparison of predicted vs ground truth poses.
    
    Args:
        preds: List of prediction tensors
        targets: List of target tensors
        movement_names: List of movement names
        output_path: Path to save the figure
    """
    num_samples = len(preds)
    time_points = [0, 10, 20, 30]  # Frames to visualize
    
    fig, axes = plt.subplots(num_samples, len(time_points), figsize=(len(time_points)*4, num_samples*3))
    
    for i in range(num_samples):
        pred = preds[i].numpy()
        target = targets[i].numpy()
        
        for j, frame in enumerate(time_points):
            if frame < pred.shape[0]:
                ax = axes[i, j] if num_samples > 1 else axes[j]
                
                # Plot ground truth and prediction for this frame
                ax.plot(range(pred.shape[1]), target[frame], 'g-', alpha=0.7, label='Ground Truth')
                ax.plot(range(pred.shape[1]), pred[frame], 'r--', alpha=0.7, label='Prediction')
                
                ax.set_title(f"{movement_names[i]}\nFrame {frame}")
                if i == 0 and j == 0:
                    ax.legend()
                
                ax.set_xticks([])
                ax.set_ylim(-3, 3)
    
    plt.suptitle("Qualitative Comparison: Ground Truth vs Prediction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def tensor_to_python(value):
    """Convert tensor values to native Python types."""
    if hasattr(value, 'item'):  # Check if it's a tensor
        try:
            return value.item()  # For scalar tensors
        except ValueError:
            return str(value)    # For non-scalar tensors
    elif isinstance(value, (list, tuple)):
        return [tensor_to_python(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): tensor_to_python(v) for k, v in value.items()}
    return value