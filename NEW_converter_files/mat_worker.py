import numpy as np
import scipy.io as sio
import os
import sys
from shadow_utils import (
    convert_row_to_shadow_angles,
    scan_dataset_for_ranges,
    set_shadow_rest_baseline
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.handAngleData import HandAnglesData

def process_mat_file(mat_path):
    try:
        mat_dir = os.path.dirname(mat_path) or '.'
        mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat")]

        has_e1 = any("_E1_" in f for f in mat_files)
        has_e2 = any("_E2_" in f for f in mat_files)
        has_e3 = any("_E3_" in f for f in mat_files)

        # 🔄 Choose baseline file smartly
        if has_e1 and has_e2:
            baseline_file = next((f for f in mat_files if "_E1_" in f), None)
            baseline_desc = "E1 (special case)"
        elif has_e2:
            baseline_file = next((f for f in mat_files if "_E2_" in f), None)
            baseline_desc = "E2"
        else:
            print(f"⚠️ Skipping {mat_path}: No E1 or E2 file found for baseline.")
            return None

        baseline_path = os.path.join(mat_dir, baseline_file)
        print(f"🔄 Computing baseline from: {baseline_desc} ({baseline_path})")

        baseline_data = sio.loadmat(baseline_path)
        if 'angles' not in baseline_data or 'restimulus' not in baseline_data:
            print(f"⚠️ Skipping {mat_path}: Baseline file missing 'angles' or 'restimulus'")
            return None

        angles = baseline_data['angles']
        movements = baseline_data['restimulus'].flatten()
        repetitions = baseline_data.get('re_repetition') or baseline_data.get('repetition')
        if repetitions is None:
            repetitions = np.zeros_like(movements)
        else:
            repetitions = repetitions.flatten()

        set_shadow_rest_baseline(angles, movements, repetitions)

        # ✅ Now process the target file
        mat_data = sio.loadmat(mat_path)
        if 'angles' not in mat_data or 'restimulus' not in mat_data:
            print(f"⚠️ Skipping {mat_path}: missing 'angles' or 'restimulus'")
            return None

        data = mat_data['angles']
        movement_data = mat_data['restimulus'].flatten()
        repetitions = mat_data.get('re_repetition') or mat_data.get('repetition')
        if repetitions is None:
            repetitions = np.zeros_like(movement_data)
        else:
            repetitions = repetitions.flatten()

        print(f"Processing {mat_path}...")

        # ✅ Scan dataset ranges
        print(f"Scanning ranges for {mat_path}...")
        scan_dataset_for_ranges(data)

        frames = []
        labels = []

        joint_names = (
            ['rh_WRJ1', 'rh_WRJ2',
             'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
             'rh_LFJ5'] +
            [f'rh_{finger}J{idx}' for finger in ['FF', 'MF', 'RF', 'LF'] for idx in range(1, 5)]
        )

        # Skip logic: determine if current file is Table B
        skip_movements_b = {4, 5, 8}
        is_table_b = False

        if "_E1_" in mat_path:
            is_table_b = True  # E1 always Table B
        elif "_E2_" in mat_path and has_e3:
            is_table_b = True  # E2 + E3 dir → E2 is Table B
        else:
            is_table_b = False  # E2 in E1+E2 dir → Table C (no skip)

        for i in range(data.shape[0]):
            row = data[i, :]
            movement = int(movement_data[i])
            rep = int(repetitions[i])

            if movement == 0 or rep == 0:
                continue  # Skip rest frames and invalid repetitions

            # Apply skip logic for Table B files
            if is_table_b and movement in skip_movements_b:
                print(f"⏭️ Skipping movement {movement} (Table B skip)")
                continue

            hand_data = convert_row_to_shadow_angles(row, movement, rep, mat_path)

            frames.append([getattr(hand_data, j) for j in joint_names])
            labels.append(movement)

        return np.array(frames), np.array(labels)

    except Exception as e:
        print(f"❌ Error in {mat_path}: {e}")
        return None
