"""
Configuration file for Hand Motion Prediction model.

This module defines constants and configuration parameters used across 
model training, evaluation, and visualization.
"""

# === Sequence parameters ===
T_OBS = 30  # Number of observed frames
T_PRED = 40  # Number of frames to predict
T_TOTAL = T_OBS + T_PRED

# === Model parameters ===
NUM_JOINTS = 24  # Total number of hand joints
MEMORY_CELLS = 16  # Memory cells for pattern storage

# === Joint mapping and groups ===
# Groups joints based on hand structure for progressive decoding
PROGRESSIVE_STAGES = [
    [0], [1],                       # Wrist (WRJ1, WRJ2)
    [2, 3, 4, 5, 6],                # Thumb (THJ1-5)
    [7, 8, 9, 10],                  # Index (FFJ1-4)
    [11, 12, 13, 14],               # Middle (MFJ1-4)
    [15, 16, 17, 18],               # Ring (RFJ1-4)
    [19, 20, 21, 22, 23]            # Little/Pinky (LFJ1-5)
]

# Joints that need special handling (masking) - Thumb joints and Little finger joint 5
MASKED_JOINTS = [2, 3, 4, 5, 6, 23]

# Shadow hand joint names
JOINT_NAMES = [
    'WRJ1', 'WRJ2',                  # Wrist
    'THJ1', 'THJ2', 'THJ3', 'THJ4', 'THJ5',  # Thumb (5 joints)
    'FFJ1', 'FFJ2', 'FFJ3', 'FFJ4',  # Index (4 joints)
    'MFJ1', 'MFJ2', 'MFJ3', 'MFJ4',  # Middle (4 joints)
    'RFJ1', 'RFJ2', 'RFJ3', 'RFJ4',  # Ring (4 joints)
    'LFJ1', 'LFJ2', 'LFJ3', 'LFJ4', 'LFJ5'  # Little/Pinky (5 joints)
]

# === Evaluation parameters ===
# Frames per second for evaluation timesteps
FPS = 25
# Standard evaluation timepoints in milliseconds
EVAL_MS = [80, 160, 320, 400, 560, 1000]
# Convert ms to frame indices
MS_TO_FRAMES = {ms: int(ms * FPS / 1000) for ms in EVAL_MS}

# === File paths ===
# Default directories for outputs
MODELS_DIR = "../models"
EVALUATIONS_DIR = "../evaluations"
PLOTS_DIR = "../plots"
DEFAULT_MODEL_PATH = f"{MODELS_DIR}/best_model.pth"