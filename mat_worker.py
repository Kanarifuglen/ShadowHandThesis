import numpy as np
import scipy.io as sio
import os
import sys
from shadow_utils import (
    build_dynamic_ranges,
    convert_row_to_shadow_angles,
    NINAPRO_MAPPING,
    SHADOW_REST
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.handAngleData import HandAnglesData

def process_mat_file(mat_path):
    try:
        mat_data = sio.loadmat(mat_path)
        if 'angles' not in mat_data or 'restimulus' not in mat_data:
            print(f"⚠️ Skipping {mat_path}: missing 'angles' or 'restimulus'")
            return None

        data = mat_data['angles']
        movement_data = mat_data['restimulus'].flatten()
        current_ranges = build_dynamic_ranges(data, movement_data)

        frames = []
        labels = []

        joint_names = [
            'rh_WRJ1', 'rh_WRJ2',
            'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
            'rh_LFJ5'
        ] + [f'rh_{finger}J{idx}' for finger in ['FF', 'MF', 'RF', 'LF'] for idx in range(1, 5)]

        rest_frames_added = 0
        for i in range(data.shape[0]):
            row = data[i, :]
            movement = movement_data[i]

            if movement == 0:
                if rest_frames_added >= 1:
                    continue  # Skip more than 1 rest frames
                rest_dict = {
                    'rh_WRJ1': float(row[NINAPRO_MAPPING['WRIST']['F']]),
                    'rh_WRJ2': float(row[NINAPRO_MAPPING['WRIST']['A']]),
                    'rh_THJ1': 0, 'rh_THJ2': 0, 'rh_THJ3': 0, 'rh_THJ4': 0, 'rh_THJ5': 0,
                    'rh_LFJ5': 0,
                }
                for finger in ['FF', 'MF', 'RF', 'LF']:
                    for idx, joint in enumerate(['DIP', 'PIP', 'MCP', 'MCP_A'], 1):
                        rest_dict[f'rh_{finger}J{idx}'] = SHADOW_REST[finger][joint]
                hand_data = HandAnglesData.fromDict(rest_dict)
                rest_frames_added += 1
            else:
                hand_data = convert_row_to_shadow_angles(row, movement, current_ranges, mat_path)

            frames.append([getattr(hand_data, j) for j in joint_names])
            labels.append(movement)

        return np.array(frames), np.array(labels)

    except Exception as e:
        print(f"❌ Error in {mat_path}: {e}")
        return None
