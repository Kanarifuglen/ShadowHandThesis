"""
Utility functions for evaluating hand motion prediction models.

This module contains functions for analyzing and visualizing model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Import configuration
from config import (
    JOINT_NAMES, EVAL_MS, 
    PLOTS_DIR, EVALUATIONS_DIR
)

def plot_per_joint_error(predictions, targets, mask, joint_names=JOINT_NAMES):
    """Enhanced version to match thesis requirements for Fig ref:joint_error."""
    joint_errors = []
    for j in range(targets.shape[-1]):
        if mask[j]:  # Skip masked joints
            joint_error = (predictions[:, :, j] - targets[:, :, j]).abs().mean().item()
            joint_errors.append((joint_names[j], joint_error))
    
    # Define joint groups for better visualization
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
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Get colors for each joint
    colors = []
    for name, _ in joint_errors:
        for group, joints in joint_groups.items():
            if name in joints:
                colors.append(group_colors[group])
                break
        else:
            colors.append('gray')
    
    # Extract data for plotting
    names, errors = zip(*joint_errors)
    
    # Plot with custom colors for each joint group
    plt.bar(names, errors, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title("Mean Error per Joint")
    plt.ylabel("Mean Absolute Error (radians)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add legend for joint groups
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in group_colors.values()]
    plt.legend(handles, group_colors.keys(), loc='upper right')
    
    plt.tight_layout()
    
    # Save with high resolution
    plt.savefig(f"{PLOTS_DIR}/per_joint_error.png", dpi=300)
    plt.close()


def plot_error_progression_by_frame(predictions, targets, mask):
    """Plot how error accumulates with each predicted frame."""
    # Calculate error at each frame
    frame_errors = []
    for t in range(predictions.shape[1]):
        frame_error = (predictions[:, t, mask] - targets[:, t, mask]).abs().mean().item()
        frame_errors.append(frame_error)
    
    # Create plot
    plt.figure(figsize=(10, 5))
    plt.plot(frame_errors, marker='o')
    plt.title("Error Progression by Frame")
    plt.xlabel("Frames into Future")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/error_progression.png")
    plt.close()


def plot_action_comparison(movement_results, eval_ms=EVAL_MS):
    """Create a grouped bar chart comparing performance across actions."""
    actions = list(movement_results.keys())
    
    # Set up plot with multiple bars per action (one for each time horizon)
    plt.figure(figsize=(14, 8))
    x = np.arange(len(actions))
    width = 0.8 / len(eval_ms)
    
    for i, ms in enumerate(eval_ms):
        values = []
        for action in actions:
            avg_error = np.mean(movement_results[action][ms]) if movement_results[action][ms] else np.nan
            values.append(avg_error)
        
        plt.bar(x + i*width - 0.4, values, width, label=f'{ms}ms')
    
    plt.xlabel('Actions')
    plt.ylabel('Mean Error')
    plt.title('Error by Action and Prediction Length')
    plt.xticks(x, actions, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/action_comparison.png")
    plt.close()


def generate_detailed_error_table(movement_results, eval_ms=EVAL_MS, 
                                 output_path=None):
    """Generate a CSV with detailed breakdown of errors by action and time horizon."""
    if output_path is None:
        os.makedirs(EVALUATIONS_DIR, exist_ok=True)
        output_path = f'{EVALUATIONS_DIR}/detailed_error_breakdown.csv'
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row with time horizons
        header = ['Movement'] + [f'{ms}ms Mean' for ms in eval_ms] + [f'{ms}ms StdDev' for ms in eval_ms]
        writer.writerow(header)
        
        # Data rows for each action
        for movement, metrics in movement_results.items():
            row = [movement]
            
            # Add mean values
            for ms in eval_ms:
                if metrics[ms]:
                    row.append(f"{sum(metrics[ms])/len(metrics[ms]):.4f}")
                else:
                    row.append('N/A')
            
            # Add standard deviation values
            for ms in eval_ms:
                if len(metrics[ms]) > 1:
                    row.append(f"{np.std(metrics[ms]):.4f}")
                else:
                    row.append('N/A')
                
            writer.writerow(row)
            
        # Add average row
        avg_row = ['Average']
        for ms in eval_ms:
            all_errors = []
            for movement in movement_results.values():
                all_errors.extend(movement[ms])
            
            if all_errors:
                avg_row.append(f"{sum(all_errors)/len(all_errors):.4f}")
                if len(all_errors) > 1:
                    avg_row.append(f"{np.std(all_errors):.4f}")
                else:
                    avg_row.append('N/A')
            else:
                avg_row.append('N/A')
                avg_row.append('N/A')
                
        writer.writerow(avg_row)
    
    print(f"Detailed error table exported to {output_path}")


def generate_joint_group_table(predictions, targets, joint_names=JOINT_NAMES):
    """Create a table analyzing performance by joint groups for the Shadow hand."""
    # Define joint groups
    joint_groups = {
        'Wrist': [0, 1],                 # WRJ1, WRJ2
        'Thumb': [2, 3, 4, 5, 6],        # THJ1-5
        'Index': [7, 8, 9, 10],          # FFJ1-4
        'Middle': [11, 12, 13, 14],      # MFJ1-4
        'Ring': [15, 16, 17, 18],        # RFJ1-4
        'Little': [19, 20, 21, 22, 23]   # LFJ1-5
    }
    
    # Create output directory
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)
    
    # Generate CSV table
    with open(f'{EVALUATIONS_DIR}/joint_group_analysis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Joint Group', 'Mean Error', 'Max Error', 'Min Error', 'Standard Deviation'])
        
        # Calculate statistics for each group
        for group_name, indices in joint_groups.items():
            group_errors = []
            
            for j in indices:
                # Skip masked joints
                if j in MASKED_JOINTS:
                    continue
                    
                joint_error = (predictions[:, :, j] - targets[:, :, j]).abs().mean().item()
                group_errors.append(joint_error)
            
            if group_errors:
                writer.writerow([
                    group_name, 
                    f"{np.mean(group_errors):.4f}",
                    f"{np.max(group_errors):.4f}",
                    f"{np.min(group_errors):.4f}",
                    f"{np.std(group_errors):.4f}"
                ])
    
    # Generate visualized version as figure
    plt.figure(figsize=(10, 6))
    
    group_means = []
    group_stds = []
    group_names = []
    
    for group_name, indices in joint_groups.items():
        group_errors = []
        
        for j in indices:
            if j not in MASKED_JOINTS:
                joint_error = (predictions[:, :, j] - targets[:, :, j]).abs().mean().item()
                group_errors.append(joint_error)
        
        if group_errors:
            group_means.append(np.mean(group_errors))
            group_stds.append(np.std(group_errors))
            group_names.append(group_name)
    
    # Sort by mean error
    sorted_indices = np.argsort(group_means)
    sorted_means = [group_means[i] for i in sorted_indices]
    sorted_stds = [group_stds[i] for i in sorted_indices]
    sorted_names = [group_names[i] for i in sorted_indices]
    
    # Plot with error bars
    plt.bar(range(len(sorted_names)), sorted_means, yerr=sorted_stds, 
            capsize=5, color='skyblue', edgecolor='navy')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45)
    plt.ylabel('Mean Absolute Error (radians)')
    plt.title('Average Error by Joint Group')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/joint_group_error.png', dpi=300)
    plt.close()
    
    print(f"Joint group analysis saved to {EVALUATIONS_DIR}/joint_group_analysis.csv")
    print(f"Joint group visualization saved to {PLOTS_DIR}/joint_group_error.png")


def analyze_dct_coefficients(input_dct, baseline_dct, pred_residual):
    """Enhanced DCT coefficient analysis for thesis Figure/Table."""
    # Create output directory
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)
    
    # Number of coefficients to analyze
    num_coeffs = min(20, input_dct.shape[1])
    
    # Calculate average magnitude across batch and joints
    input_mag = input_dct[:, :num_coeffs, :].abs().mean(dim=(0, 2)).cpu().numpy()
    baseline_mag = baseline_dct[:, :num_coeffs, :].abs().mean(dim=(0, 2)).cpu().numpy()
    residual_mag = pred_residual[:, :num_coeffs, :].abs().mean(dim=(0, 2)).cpu().numpy()
    
    # Save in CSV format for paper table
    with open(f'{EVALUATIONS_DIR}/dct_coefficient_analysis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Coefficient Index', 'Input Magnitude', 'Baseline Magnitude', 'Residual Magnitude'])
        
        for i in range(num_coeffs):
            writer.writerow([
                i, 
                f"{input_mag[i]:.6f}", 
                f"{baseline_mag[i]:.6f}", 
                f"{residual_mag[i]:.6f}"
            ])
    
    # Create visualization for thesis
    plt.figure(figsize=(10, 6))
    
    # Plot each line with different style
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
    plt.savefig(f'{PLOTS_DIR}/dct_coefficient_analysis.png', dpi=300)
    plt.close()
    
    print(f"DCT coefficient analysis exported to {EVALUATIONS_DIR}/dct_coefficient_analysis.csv")
    print(f"DCT coefficient visualization saved to {PLOTS_DIR}/dct_coefficient_analysis.png")

def visualize_predictions(pred, target, sample_idx, movement_name, num_joints):
    """Visualize predictions in a standardized format.
    
    Args:
        pred: Predicted poses tensor
        target: Ground truth poses tensor
        sample_idx: Sample index for file naming
        movement_name: Name of the movement for title
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Ensure directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Plot trajectory of key joints over time
    plt.figure(figsize=(15, 10))
    
    # Select representative joints including one from each finger
    key_joints = [0, 6, 10, 14, 18, 23]  # Wrist, Thumb tip, Index tip, Middle tip, Ring tip, Little tip
    joint_labels = ["Wrist (WRJ1)", "Thumb tip (THJ5)", "Index tip (FFJ4)", 
                   "Middle tip (MFJ4)", "Ring tip (RFJ4)", "Little tip (LFJ5)"]
    
    for j_idx, joint_idx in enumerate(key_joints):
        plt.subplot(len(key_joints), 1, j_idx + 1)
        plt.plot(target_np[:, joint_idx], 'g-', label='Ground Truth')
        plt.plot(pred_np[:, joint_idx], 'r--', label='Prediction')
        plt.title(f"{joint_labels[j_idx]} Trajectory")
        if j_idx == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/joint_trajectories_{sample_idx}.png")
    plt.close()
    
    # 2. Plot poses at specific time horizons (80ms, 160ms, 400ms, 1000ms)
    horizon_frames = [2, 4, 10, 25]  # Convert ms to frames (assuming 25fps)
    horizon_labels = ["80ms", "160ms", "400ms", "1000ms"]
    
    plt.figure(figsize=(16, 6))
    plt.suptitle(f"Prediction: {movement_name}", fontsize=16)
    
    for h_idx, frame_idx in enumerate(horizon_frames):
        if frame_idx < pred_np.shape[0]:
            plt.subplot(1, len(horizon_frames), h_idx + 1)
            
            # Create scatter plot for each joint position
            plt.scatter(range(num_joints), target_np[frame_idx], c='g', label='Ground Truth')
            plt.scatter(range(num_joints), pred_np[frame_idx], c='r', marker='x', label='Prediction')
            
            plt.title(f"t+{horizon_labels[h_idx]}")
            if h_idx == 0:
                plt.legend()
            plt.ylim(-3, 3)  # Standardize y-axis for better comparison
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/horizon_comparison_{sample_idx}.png")
    plt.close()