# shadow_utils.py (Updated: Handle MCP Extension)

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.handAngleData import HandAnglesData

NINAPRO_MAPPING = {
    'WRIST': {'F': 20, 'A': 21},
    'FF': {'MCP': 4, 'PIP': 6, 'DIP': 16},
    'MF': {'MCP': 7, 'PIP': 8, 'DIP': 17},
    'RF': {'MCP': 9, 'PIP': 11, 'DIP': 18},
    'LF': {'MCP': 13, 'PIP': 15, 'DIP': 19},
    'LF_EXTRA': {'CMC': 12}
}

# Shadow hand joint limits (degrees)
SHADOW_ANGLE_LIMITS = {
    'MCP': (-15, 90),  # Includes extension
    'PIP': (0, 90),
    'DIP': (0, 90)
}

JOINT_TUNING = {
    'FF': {'DIP': 1.2, 'PIP': 1.0, 'MCP': 1.0},
    'MF': {'DIP': 1.2, 'PIP': 0.6, 'MCP': 1.25},
    'RF': {'DIP': 1.0, 'PIP': 0.9, 'MCP': 0.85},
    'LF': {'DIP': 0.8, 'PIP': 0.75, 'MCP': 0.7},
}


baseline = {}
GLOBAL_MIN_MAX = {}

def set_baseline_from_first_row(row):
    """Initialize baseline from the first frame (used as rest pose)."""
    for finger in ['FF', 'MF', 'RF', 'LF']:
        baseline[finger] = {}
        for joint in ['MCP', 'PIP', 'DIP']:
            col_idx = NINAPRO_MAPPING[finger][joint]
            baseline[finger][joint] = float(row[col_idx])
    print("✅ Baseline (rest pose) set:", baseline)

def scan_dataset_for_ranges(angles, movements):
    """Scan dataset to find per-file min/max for each finger/joint, skipping movement 0."""
    global GLOBAL_MIN_MAX

    # Initialize with very large/small numbers
    GLOBAL_MIN_MAX = {finger: {joint: [np.inf, -np.inf] for joint in ['MCP', 'PIP', 'DIP']}
                      for finger in ['FF', 'MF', 'RF', 'LF']}

    for i in range(angles.shape[0]):
        movement_id = int(movements[i])
        if movement_id == 0:
            continue  # Skip rest frames entirely

        row = angles[i]
        for finger in ['FF', 'MF', 'RF', 'LF']:
            for joint in ['MCP', 'PIP', 'DIP']:
                col_idx = NINAPRO_MAPPING[finger][joint]
                val = float(row[col_idx])
                min_val, max_val = GLOBAL_MIN_MAX[finger][joint]
                # Update min/max
                GLOBAL_MIN_MAX[finger][joint][0] = min(min_val, val)
                GLOBAL_MIN_MAX[finger][joint][1] = max(max_val, val)

    print("✅ Scanned min/max per finger + joint:")
    for finger in GLOBAL_MIN_MAX:
        print(f"{finger}: {GLOBAL_MIN_MAX[finger]}")

def normalize_relative(val, min_val, max_val):
    """Normalize a value between min/max to [0, 1]."""
    return np.clip((val - min_val) / (max_val - min_val), 0, 1)


def scale_to_shadow(norm_val, joint):
    """Scale normalized value to shadow hand range."""
    shadow_min, shadow_max = SHADOW_ANGLE_LIMITS[joint]
    return norm_val * (shadow_max - shadow_min) + shadow_min

def convert_row_to_shadow_angles(row, movement_id, current_ranges, source_file):
    """Convert a single row of angles into Shadow Hand angles."""
    joint_dict = {
        'rh_WRJ1': float(row[NINAPRO_MAPPING['WRIST']['F']]),
        'rh_WRJ2': float(row[NINAPRO_MAPPING['WRIST']['A']]),
        'rh_THJ1': 0, 'rh_THJ2': 0, 'rh_THJ3': 0, 'rh_THJ4': 0, 'rh_THJ5': 0
    }

    for finger in ['FF', 'MF', 'RF', 'LF']:
        try:
            relative_angles = {}
            for joint in ['MCP', 'PIP', 'DIP']:
                col_idx = NINAPRO_MAPPING[finger][joint]
                raw_val = float(row[col_idx])
                baseline_val = baseline[finger][joint]
                relative = raw_val - baseline_val
                relative_angles[joint] = relative

            limits = GLOBAL_MIN_MAX[finger]
            mcp_norm = normalize_relative(relative_angles['MCP'], *limits['MCP'])
            pip_norm = normalize_relative(relative_angles['PIP'], *limits['PIP'])
            dip_norm = normalize_relative(relative_angles['DIP'], *limits['DIP'])

            # Scale to shadow hand ranges
            mcp_shadow = scale_to_shadow(mcp_norm, 'MCP') * JOINT_TUNING[finger]['MCP']
            pip_shadow = scale_to_shadow(pip_norm, 'PIP') * JOINT_TUNING[finger]['PIP']
            dip_shadow = scale_to_shadow(dip_norm, 'DIP') * JOINT_TUNING[finger]['DIP']


            joint_dict[f'rh_{finger}J1'] = dip_shadow
            joint_dict[f'rh_{finger}J2'] = pip_shadow
            joint_dict[f'rh_{finger}J3'] = mcp_shadow
            joint_dict[f'rh_{finger}J4'] = 0.0  # Adduction unused

        except Exception as e:
            print(f"⚠️ Error mapping finger {finger} in {source_file}: {e}")
            for idx in range(1, 5):
                joint_dict[f'rh_{finger}J{idx}'] = 0.0

    joint_dict['rh_LFJ5'] = float(row[NINAPRO_MAPPING['LF_EXTRA']['CMC']])
    return HandAnglesData.fromDict(joint_dict)


def get_baseline_hand_data():
    """
    Returns a HandAnglesData object representing the baseline (rest pose),
    using the current baseline angles.
    """
    joint_dict = {
        'rh_WRJ1': 0,
        'rh_WRJ2': 0,
        'rh_THJ1': 0, 'rh_THJ2': 0, 'rh_THJ3': 0, 'rh_THJ4': 0, 'rh_THJ5': 0
    }

    # Fill each finger's MCP/PIP/DIP from baseline
    for finger in ['FF', 'MF', 'RF', 'LF']:
        joint_dict[f'rh_{finger}J1'] = baseline[finger]['DIP']
        joint_dict[f'rh_{finger}J2'] = baseline[finger]['PIP']
        joint_dict[f'rh_{finger}J3'] = baseline[finger]['MCP']
        joint_dict[f'rh_{finger}J4'] = 0.0  # Adduction unused

    joint_dict['rh_LFJ5'] = 0.0  # Optional thumb/fifth joint reset
    return HandAnglesData.fromDict(joint_dict)
