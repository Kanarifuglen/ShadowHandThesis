# shadow_utils.py (Global Anatomical Normalization)

import numpy as np
import sys
import os
import scipy.interpolate as interp

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

SHADOW_ANGLE_LIMITS = {
    'MCP': (-15, 90),
    'PIP': (0, 90),
    'DIP': (0, 90)
}

JOINT_TUNING = {
    'FF': {'DIP': 1.0, 'PIP': 1.0, 'MCP': 1.0},
    'MF': {'DIP': 1.0, 'PIP': 1.0, 'MCP': 1.0},
    'RF': {'DIP': 1.0, 'PIP': 1.0, 'MCP': 1.0},
    'LF': {'DIP': 1.0, 'PIP': 1.0, 'MCP': 1.0},
}

SHADOW_REST_BASELINE = {}

ANATOMICAL_MIN_MAX = {
    finger: {joint: [np.inf, -np.inf] for joint in ['MCP', 'PIP', 'DIP']}
    for finger in ['FF', 'MF', 'RF', 'LF']
}

E2_EXTENSION= set(range(9, 17))

MOVEMENT_NAMES_B = {
    1: "Thumb up", 2: "Extension of index and middle, flexion of the others",
    3: "Flexion of ring and little finger, extension of the others",
    4: "Thumb opposing base of little finger", 5: "Abduction of all fingers",
    6: "Fingers flexed together into a fist", 7: "Pointing index", 8: "Adduction on extended fingers",
    9: "Wrist supination (MF axis)", 10: "Wrist supination", 11: "Wrist supination (LF axis)",
    12: "Wrist pronation (LF axis)", 13: "Wrist flexion", 14: "Wrist extension",
    15: "Wrist radial deviation", 16: "Wrist ulnar deviation", 17: "Wrist extension with closed hand", 0: "Rest"
}

MOVEMENT_NAMES_C = {
    1: "Large diameter grasp", 2: "Small diameter grasp", 3: "Fixed hook grasp", 4: "Index finger extension grasp",
    5: "Medium wrap", 6: "Ring grasp", 7: "Prismatic four fingers grasp", 8: "Stick grasp", 9: "Writing tripod grasp",
    10: "Power sphere grasp", 11: "Three finger sphere grasp", 12: "Precision sphere grasp", 13: "Tripod grasp",
    14: "Prismatic pinch grasp", 15: "Tip pinch grasp", 16: "Quadpod grasp", 17: "Lateral grasp",
    18: "Parallel extension grasp", 19: "Extension type grasp", 20: "Power disk grasp",
    21: "Open a bottle with a tripod grasp", 22: "Turn a screw", 23: "Cut something", 0: "Rest"
}


def set_shadow_rest_baseline(angles, movements, repetitions):
    global SHADOW_REST_BASELINE
    idx_movement_11 = np.where(movements == 11)[0]
    movement_11_reps = np.unique(repetitions[idx_movement_11])

    rest_values = {finger: {joint: [] for joint in ['MCP', 'PIP', 'DIP']} for finger in ['FF', 'MF', 'RF', 'LF']}

    for rep in movement_11_reps:
        rep_mask = repetitions[idx_movement_11].squeeze() == rep
        rep_indices = idx_movement_11[rep_mask]

        if len(rep_indices) == 0:
            continue

        start = rep_indices[0]
        end = rep_indices[-1]
        center_start = start + (end - start) // 3
        center_end = end - (end - start) // 3

        center_rows = angles[center_start:center_end]

        for finger in ['FF', 'MF', 'RF', 'LF']:
            for joint in ['MCP', 'PIP', 'DIP']:
                col_idx = NINAPRO_MAPPING[finger][joint]
                mean_val = np.mean(center_rows[:, col_idx])
                rest_values[finger][joint].append(mean_val)

    SHADOW_REST_BASELINE = {
        finger: {joint: np.mean(rest_values[finger][joint]) for joint in rest_values[finger]}
        for finger in rest_values
    }

    #print("\n✅ Computed SHADOW_REST_BASELINE from Movement 11 center regions:")
    #for finger in SHADOW_REST_BASELINE:
       # print(f"  {finger}: {SHADOW_REST_BASELINE[finger]}")

def scan_dataset_for_ranges(angles):
    global ANATOMICAL_MIN_MAX
    for row in angles:
        for finger in ['FF', 'MF', 'RF', 'LF']:
            for joint in ['MCP', 'PIP', 'DIP']:
                col_idx = NINAPRO_MAPPING[finger][joint]
                val = float(row[col_idx])
                min_val, max_val = ANATOMICAL_MIN_MAX[finger][joint]
                ANATOMICAL_MIN_MAX[finger][joint][0] = min(min_val, val)
                ANATOMICAL_MIN_MAX[finger][joint][1] = max(max_val, val)
    #print("\n✅ Scanned GLOBAL anatomical ranges (absolute):")
    #for finger in ANATOMICAL_MIN_MAX:
    #    print(f"{finger}: {ANATOMICAL_MIN_MAX[finger]}")

def scale_to_shadow(norm_val, joint):
    shadow_min, shadow_max = SHADOW_ANGLE_LIMITS[joint]
    return norm_val * (shadow_max - shadow_min) + shadow_min

E2_EXTENSION_STANDARD = set(range(9, 17))  # 9–16 (standard E2)
E2_EXTENSION_OFFSET = set(m + 17 for m in range(9, 17))  # 26–33 (E2 in E1+E2 dirs)

E2_EXTENSION_STANDARD = set(range(9, 17))  # 9–16 (wrist-only movements in Table B)

def convert_row_to_shadow_angles(row, movement_id, rep_id, source_file):
    joint_dict = {
        'rh_WRJ1': float(row[NINAPRO_MAPPING['WRIST']['F']]),
        'rh_WRJ2': float(row[NINAPRO_MAPPING['WRIST']['A']]),
        'rh_THJ1': 0, 'rh_THJ2': 0, 'rh_THJ3': 0, 'rh_THJ4': 0, 'rh_THJ5': 0
    }

    # Determine if we are processing a Table B file (E1 or E2 in E2+E3 dirs)
    is_table_b = False

    if "_E1_" in source_file:
        is_table_b = True  # E1 is always Table B
    elif "_E2_" in source_file:
        source_dir = os.path.dirname(source_file) or '.'
        mat_files = [f for f in os.listdir(source_dir) if f.endswith(".mat")]
        has_e3 = any("_E3_" in f for f in mat_files)
        if has_e3:
            is_table_b = True  # E2 + E3 dir → Table B in E2, Table C in E3
        else:
            is_table_b = False  # E1 + E2 dir → Table B in E1, Table C in E2

    for finger in ['FF', 'MF', 'RF', 'LF']:
        try:
            # Table B file + movement in 9–16 (wrist-only movements)
            if is_table_b and movement_id in E2_EXTENSION_STANDARD:
                joint_dict[f'rh_{finger}J1'] = SHADOW_REST_BASELINE[finger]['DIP']
                joint_dict[f'rh_{finger}J2'] = SHADOW_REST_BASELINE[finger]['PIP']
                joint_dict[f'rh_{finger}J3'] = SHADOW_REST_BASELINE[finger]['MCP']
                joint_dict[f'rh_{finger}J4'] = 0.0
                #print(f"🔧 Overriding finger {finger} to baseline for movement {movement_id} (Table B wrist-only)")
                continue

            if movement_id == 0:
                joint_dict[f'rh_{finger}J1'] = SHADOW_REST_BASELINE[finger]['DIP']
                joint_dict[f'rh_{finger}J2'] = SHADOW_REST_BASELINE[finger]['PIP']
                joint_dict[f'rh_{finger}J3'] = SHADOW_REST_BASELINE[finger]['MCP']
                joint_dict[f'rh_{finger}J4'] = 0.0
            else:
                for joint in ['MCP', 'PIP', 'DIP']:
                    col_idx = NINAPRO_MAPPING[finger][joint]
                    raw_val = float(row[col_idx])
                    if np.isnan(raw_val):
                        raw_val = 0.0
                        #print(f"🚨 NaN raw_val detected for {finger} {joint} in {source_file}, movement {movement_id}, rep {rep_id}, col_idx {col_idx}")


                    anat_min, anat_max = ANATOMICAL_MIN_MAX[finger][joint]
                    if np.isinf(anat_min) or np.isinf(anat_max):
                        print(f"🚨 Invalid anat_min/anat_max for {finger} {joint} in {source_file} "
                            f"(anat_min: {anat_min}, anat_max: {anat_max}). Replacing with defaults.")
                        anat_min, anat_max = 0.0, 90.0  # Or whatever default range makes sense
                    effective_range = anat_max - anat_min if anat_min != anat_max else 10.0

                    norm = (raw_val - anat_min) / effective_range
                    norm = np.clip(norm, 0, 1)

                    shadow_scaled = scale_to_shadow(norm, joint) * JOINT_TUNING[finger][joint]

                    joint_name = {
                        'DIP': f'rh_{finger}J1',
                        'PIP': f'rh_{finger}J2',
                        'MCP': f'rh_{finger}J3'
                    }[joint]

                    joint_dict[joint_name] = shadow_scaled
                    if any(np.isnan(val) for val in joint_dict.values()):
                       # print(joint_dict.values())
                        print(f"🚨 NaNs found in final joint_dict for {source_file} | movement {movement_id}, rep {rep_id}")


                    #print(f"{finger} | {joint}: raw={raw_val:.2f}, anat_range=({anat_min:.2f}, {anat_max:.2f}), "
                    #      f"norm={norm:.2f}, scaled={shadow_scaled:.2f}")

                joint_dict[f'rh_{finger}J4'] = 0.0

        except Exception as e:
            print(f"⚠️ Error mapping finger {finger} in {source_file}: {e}")
            for idx in range(1, 5):
                joint_dict[f'rh_{finger}J{idx}'] = 0.0

    joint_dict['rh_LFJ5'] = 0
    return HandAnglesData.fromDict(joint_dict)



def get_baseline_hand_data():
    joint_dict = {
        'rh_WRJ1': 0,
        'rh_WRJ2': 0,
        'rh_THJ1': 0, 'rh_THJ2': 0, 'rh_THJ3': 0, 'rh_THJ4': 0, 'rh_THJ5': 0
    }

    for finger in ['FF', 'MF', 'RF', 'LF']:
        joint_dict[f'rh_{finger}J1'] = SHADOW_REST_BASELINE[finger]['DIP']
        joint_dict[f'rh_{finger}J2'] = SHADOW_REST_BASELINE[finger]['PIP']
        joint_dict[f'rh_{finger}J3'] = SHADOW_REST_BASELINE[finger]['MCP']
        joint_dict[f'rh_{finger}J4'] = 0.0

    joint_dict['rh_LFJ5'] = 0.0
    return HandAnglesData.fromDict(joint_dict)


def resample_sequence(sequence, target_frames=130):
    """
    Resample a 2D numpy array (frames x features) to target_frames using interpolation.
    """
    num_frames, num_features = sequence.shape
    if num_frames == target_frames:
        return sequence  # already correct length
    x_old = np.linspace(0, 1, num_frames)
    x_new = np.linspace(0, 1, target_frames)
    resampled = np.zeros((target_frames, num_features))
    for f in range(num_features):
        interpolator = interp.interp1d(x_old, sequence[:, f], kind='linear')
        resampled[:, f] = interpolator(x_new)
    return resampled
