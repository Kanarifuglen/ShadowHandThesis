import math
import numpy as np
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.handAngleData import HandAnglesData

# === Constants ===

JOINT_LIMITS = {
    'MCP': (-15, 90),
    'PIP': (0, 90),
    'DIP': (0, 90),
    'MCP_A': (-20, 20),
}

NINAPRO_TRUE_RANGES = {
    'FF': {
        'DIP': (-10.42, 143.96),
        'PIP': (-13.68, 107.16),
        'MCP': (-35.98, 61.70),
        'MCP_A': (float('nan'), float('nan')),
    },
    'MF': {
        'DIP': (0, 70),
        'PIP': (0, 65),
        'MCP': (0, 20),
        'MCP_A': (-1.75, 2.00),
    },
    'RF': {
        'DIP': (-144.44, -10),
        'PIP': (-10.24, 117.62),
        'MCP': (-30.16, 50.04),
        'MCP_A': (-2.06, 9.91),
    },
    'LF': {
        'DIP': (-57.67, 25.49),
        'PIP': (-9.66, 111.52),
        'MCP': (-9.98, 144.66),
        'MCP_A': (-11.86, -8.57),
    },
}

NINAPRO_MAPPING = {
    'FF': {'MCP': 4, 'PIP': 6, 'DIP': 16, 'MCP_A': 5},
    'MF': {'MCP': 7, 'PIP': 8, 'DIP': 17, 'MCP_A': 10},
    'RF': {'MCP': 9, 'PIP': 11, 'DIP': 18, 'MCP_A': 12},
    'LF': {'MCP': 13, 'PIP': 15, 'DIP': 19, 'MCP_A': 14},
    'WRIST': {'F': 20, 'A': 21},
}

SHADOW_REST = {f: {j: 0.0 for j in JOINT_LIMITS.keys()} for f in ['FF', 'MF', 'RF', 'LF']}

# === Functions ===

def build_dynamic_ranges(data, movement_data):
    dynamic_ranges = {}
    for finger in ['FF', 'MF', 'RF', 'LF']:
        dynamic_ranges[finger] = {}
        for joint in ['DIP', 'PIP', 'MCP', 'MCP_A']:
            col = NINAPRO_MAPPING[finger][joint]
            relevant_vals = data[(movement_data > 0), col]
            relevant_vals = relevant_vals[~np.isnan(relevant_vals)]

            if len(relevant_vals) > 0:
                dynamic_min, dynamic_max = np.min(relevant_vals), np.max(relevant_vals)
                global_min, global_max = NINAPRO_TRUE_RANGES[finger][joint]

                if (finger == 'MF' and joint == 'DIP') or (finger == 'RF' and joint in ['MCP', 'PIP', 'DIP']):
                    dynamic_ranges[finger][joint] = (global_min, global_max)
                else:
                    combined_min = min(global_min, dynamic_min)
                    combined_max = max(global_max, dynamic_max)
                    dynamic_ranges[finger][joint] = (combined_min, combined_max)
            else:
                dynamic_ranges[finger][joint] = NINAPRO_TRUE_RANGES[finger][joint]
    return dynamic_ranges


def scale_relative_to_shadow(finger, joint, value, current_ranges):
    raw_min, raw_max = current_ranges[finger][joint]
    shadow_min, shadow_max = JOINT_LIMITS[joint]

    if raw_max - raw_min < 1e-6:
        return shadow_min

    value = np.clip(value, raw_min, raw_max)
    scaled = (value - raw_min) / (raw_max - raw_min)
    angle = scaled * (shadow_max - shadow_min) + shadow_min

    return np.clip(angle, shadow_min, shadow_max)


def apply_dip_constraint(pip, mcp, dip):
    if math.isnan(dip): dip = 0
    if not math.isnan(pip) and not math.isnan(mcp):
        if pip < 30 and mcp < 30:
            dip = min(dip, pip * 0.66)
        if pip < 10 and mcp < 10:
            dip = min(dip, 5)
    return max(dip, -5)


def convert_row_to_shadow_angles(row, movement, current_ranges, filename=None):
    modified_row = row.copy()

    FULLY_EXTENDED_MOVEMENTS = {4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    if filename and re.match(r'^S\d+_E2_A1\.mat$', os.path.basename(filename)) and movement in FULLY_EXTENDED_MOVEMENTS:
        for finger in ['FF', 'MF', 'RF', 'LF']:
            for joint in ['DIP', 'PIP', 'MCP', 'MCP_A']:
                modified_row[NINAPRO_MAPPING[finger][joint]] = 0
        return HandAnglesData.fromDict({
            **{f"rh_{finger}J{idx}": 0 for finger in ['FF', 'MF', 'RF', 'LF'] for idx in range(1, 5)},
            'rh_WRJ1': float(modified_row[NINAPRO_MAPPING['WRIST']['F']]),
            'rh_WRJ2': float(modified_row[NINAPRO_MAPPING['WRIST']['A']]),
            'rh_THJ1': 0, 'rh_THJ2': 0, 'rh_THJ3': 0, 'rh_THJ4': 0, 'rh_THJ5': 0,
            'rh_LFJ5': 0,
        })

    ALIGNMENT_MOVEMENTS = {1, 2, 3, 5, 7, 8}
    if filename and re.match(r'^S\d+_E3_A1\.mat$', os.path.basename(filename)) and movement in ALIGNMENT_MOVEMENTS:
        joint_dict = {
            'rh_WRJ1': float(modified_row[NINAPRO_MAPPING['WRIST']['F']]),
            'rh_WRJ2': float(modified_row[NINAPRO_MAPPING['WRIST']['A']]),
            'rh_THJ1': 0, 'rh_THJ2': 0, 'rh_THJ3': 0, 'rh_THJ4': 0, 'rh_THJ5': 0,
            'rh_LFJ5': 0,
        }

        mf_angles = {}
        for idx, joint in enumerate(['DIP', 'PIP', 'MCP'], 1):
            value = modified_row[NINAPRO_MAPPING['FF'][joint]]
            angle = 0.0 if math.isnan(value) else scale_relative_to_shadow('FF', joint, value, current_ranges)
            mf_angles[idx] = angle

        for finger in ['FF', 'MF', 'RF', 'LF']:
            joint_dict[f'rh_{finger}J1'] = mf_angles[1]
            joint_dict[f'rh_{finger}J2'] = mf_angles[2]
            joint_dict[f'rh_{finger}J3'] = mf_angles[3]
            joint_dict[f'rh_{finger}J4'] = 0.0

        return HandAnglesData.fromDict(joint_dict)

    for finger in ['FF', 'MF', 'RF', 'LF']:
        mcp_val = modified_row[NINAPRO_MAPPING[finger]['MCP']]
        pip_val = modified_row[NINAPRO_MAPPING[finger]['PIP']]
        dip_val = modified_row[NINAPRO_MAPPING[finger]['DIP']]

        if not math.isnan(pip_val):
            if math.isnan(dip_val) or dip_val > pip_val:
                modified_row[NINAPRO_MAPPING[finger]['DIP']] = pip_val
            modified_row[NINAPRO_MAPPING[finger]['DIP']] = min(modified_row[NINAPRO_MAPPING[finger]['DIP']], pip_val * 0.6)

        if not math.isnan(mcp_val) and mcp_val < 5:
            if not math.isnan(pip_val) and pip_val > 30:
                modified_row[NINAPRO_MAPPING[finger]['PIP']] = 30
            if not math.isnan(dip_val) and dip_val > 20:
                modified_row[NINAPRO_MAPPING[finger]['DIP']] = 20

        if not math.isnan(dip_val) and dip_val < -5:
            modified_row[NINAPRO_MAPPING[finger]['DIP']] = -5

    joint_dict = {
        'rh_WRJ1': float(modified_row[NINAPRO_MAPPING['WRIST']['F']]),
        'rh_WRJ2': float(modified_row[NINAPRO_MAPPING['WRIST']['A']]),
        'rh_THJ1': 0, 'rh_THJ2': 0, 'rh_THJ3': 0, 'rh_THJ4': 0, 'rh_THJ5': 0,
        'rh_LFJ5': 0,
    }

    for finger in ['FF', 'MF', 'RF', 'LF']:
        mcp_val = modified_row[NINAPRO_MAPPING[finger]['MCP']]
        pip_val = modified_row[NINAPRO_MAPPING[finger]['PIP']]
        dip_val = modified_row[NINAPRO_MAPPING[finger]['DIP']]

        if mcp_val > 75 and finger == 'MF':
            if pip_val < 60:
                pip_val = 60
            if dip_val < 45:
                dip_val = 45
            modified_row[NINAPRO_MAPPING[finger]['PIP']] = pip_val
            modified_row[NINAPRO_MAPPING[finger]['DIP']] = dip_val

        dip_val = apply_dip_constraint(pip_val, mcp_val, dip_val)
        modified_row[NINAPRO_MAPPING[finger]['DIP']] = dip_val

        for idx, joint in enumerate(['DIP', 'PIP', 'MCP', 'MCP_A'], 1):
            raw_val = modified_row[NINAPRO_MAPPING[finger][joint]]
            scaled = 0.0 if math.isnan(raw_val) else scale_relative_to_shadow(finger, joint, raw_val, current_ranges)
            joint_dict[f'rh_{finger}J{idx}'] = 0.0 if joint == 'MCP_A' else scaled

    return HandAnglesData.fromDict(joint_dict)
