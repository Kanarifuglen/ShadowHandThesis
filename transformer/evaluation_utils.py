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
    """Create a bar chart showing error for each individual joint."""
    joint_errors = []
    for j in range(targets.shape[-1]):
        if mask[j]:  # Skip masked joints (thumb joints and LFJ5)
            joint_error = (predictions[:, :, j] - targets[:, :, j]).abs().mean().item()
            joint_errors.append((joint_names[j], joint_error))
    
    # Sort by error value
    joint_errors.sort(key=lambda x: x[1], reverse=True)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    names, errors = zip(*joint_errors)
    plt.bar(names, errors)
    plt.xticks(rotation=45, ha='right')
    plt.title("Mean Error per Joint")
    plt.ylabel("Mean Absolute Error (radians)")
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/per_joint_error.png")
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
    # Define joint groups for Shadow Hand
    joint_groups = {
        'Wrist': [0, 1],                 # WRJ1, WRJ2
        'Thumb': [2, 3, 4, 5, 6],        # THJ1-5
        'Index': [7, 8, 9, 10],          # FFJ1-4
        'Middle': [11, 12, 13, 14],      # MFJ1-4
        'Ring': [15, 16, 17, 18],        # RFJ1-4
        'Little': [19, 20, 21, 22, 23]   # LFJ1-5
    }
    
    # Ensure directory exists
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)
    
    with open(f'{EVALUATIONS_DIR}/joint_group_analysis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Joint Group', 'Mean Error', 'Max Error', 'Min Error'])
        
        for group_name, indices in joint_groups.items():
            group_errors = []
            
            for j in indices:
                joint_error = (predictions[:, :, j] - targets[:, :, j]).abs().mean().item()
                group_errors.append(joint_error)
            
            if group_errors:
                writer.writerow([
                    group_name, 
                    f"{sum(group_errors)/len(group_errors):.4f}",
                    f"{max(group_errors):.4f}",
                    f"{min(group_errors):.4f}"
                ])
    
    print(f"Joint group analysis exported to {EVALUATIONS_DIR}/joint_group_analysis.csv")


def analyze_dct_coefficients(input_dct, baseline_dct, pred_residual):
    """Create a table analyzing the importance of different DCT coefficients."""
    # Ensure directory exists
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)
    
    with open(f'{EVALUATIONS_DIR}/dct_coefficient_analysis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Coefficient Index', 'Mean Magnitude in Input', 'Mean Magnitude in Baseline', 'Mean Magnitude in Residual'])
        
        for i in range(input_dct.shape[1]):
            input_mag = input_dct[:, i, :].abs().mean().item()
            baseline_mag = baseline_dct[:, i, :].abs().mean().item() if i < baseline_dct.shape[1] else 0
            residual_mag = pred_residual[:, i, :].abs().mean().item() if i < pred_residual.shape[1] else 0
            
            writer.writerow([i, f"{input_mag:.6f}", f"{baseline_mag:.6f}", f"{residual_mag:.6f}"])
    
    print(f"DCT coefficient analysis exported to {EVALUATIONS_DIR}/dct_coefficient_analysis.csv")


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