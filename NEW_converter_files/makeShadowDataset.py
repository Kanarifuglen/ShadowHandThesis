import os
import numpy as np
import scipy.io as sio
from shadow_utils import (
    scan_dataset_for_ranges,
    convert_row_to_shadow_angles,
    set_shadow_rest_baseline,
    downsample_indices,
    MOVEMENT_NAMES_B,
    MOVEMENT_NAMES_C
)

def generate_shadow_dataset(mat_filename, output_filename="shadow_dataset.npz"):
    mat_dir = os.path.dirname(mat_filename) or '.'
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat")]

    has_e1 = any("_E1_" in f for f in mat_files)
    has_e2 = any("_E2_" in f for f in mat_files)
    has_e3 = any("_E3_" in f for f in mat_files)

    # 🔄 Choose baseline file smartly
    if has_e1 and has_e2:
        baseline_file = next((f for f in mat_files if "_E1_" in f), None)
    elif has_e2:
        baseline_file = next((f for f in mat_files if "_E2_" in f), None)
    else:
        raise FileNotFoundError(f"No E1 or E2 file found in {mat_dir} to compute baseline!")

    baseline_path = os.path.join(mat_dir, baseline_file)
    baseline_data = sio.loadmat(baseline_path)
    if 'angles' not in baseline_data or 'restimulus' not in baseline_data:
        raise ValueError(f"Baseline file {baseline_path} missing 'angles' or 'restimulus'")

    angles = baseline_data['angles']
    movements = baseline_data['restimulus'].flatten()
    repetitions = baseline_data.get('re_repetition') or baseline_data.get('repetition')
    if repetitions is None:
        repetitions = np.zeros_like(movements)
    else:
        repetitions = repetitions.flatten()

    set_shadow_rest_baseline(angles, movements, repetitions)

    # 🔍 Load the actual file to process
    mat_data = sio.loadmat(mat_filename)
    if 'angles' not in mat_data or 'restimulus' not in mat_data:
        raise ValueError("Missing 'angles' or 'restimulus' in .mat file.")

    data = mat_data['angles']
    movement_data = mat_data['restimulus'].flatten()
    repetition = mat_data.get('re_repetition') or mat_data.get('repetition')
    if repetition is None:
        repetition = np.zeros_like(movement_data)
    else:
        repetition = repetition.flatten()

    scan_dataset_for_ranges(data)

    # Metadata holders
    sequences = []
    movement_ids = []
    movement_names = []
    subject_ids = []
    exercise_tables = []
    exercise_ids = []
    session_ids = []
    valid_lengths = []

    joint_names = [
        'rh_WRJ1', 'rh_WRJ2',
        'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
        'rh_LFJ5'
    ] + [f'rh_{finger}J{idx}' for finger in ['FF', 'MF', 'RF', 'LF'] for idx in range(1, 5)]

    # Skip logic: determine if current file is Table B
    skip_movements_b = {4, 5, 8}
    is_table_b = False

    if "_E1_" in mat_filename:
        is_table_b = True  # E1 always Table B
    elif "_E2_" in mat_filename and has_e3:
        is_table_b = True  # E2 + E3 dir → E2 is Table B
    else:
        is_table_b = False  # E2 in E1+E2 dir → Table C

    # Extract session + subject metadata
    session_id = os.path.splitext(os.path.basename(mat_filename))[0]  # e.g., S1_E2_A1
    subject_id = session_id.split("_")[0]  # e.g., S1
    exercise_id = session_id.split("_")[1]  # e.g., E2
    exercise_table = 'B' if is_table_b else 'C'

    j = 0
    while j < len(movement_data):
        movement = int(movement_data[j])
        rep = int(repetition[j]) if repetition is not None else 0

        if movement == 0 or rep == 0:
            j += 1
            continue

        start = j
        while j < len(movement_data) and movement_data[j] == movement and repetition[j] == rep:
            j += 1
        end = j

        if is_table_b and movement in skip_movements_b:
            continue

        # Downsample this repetition
        num_frames = end - start
        indices = downsample_indices(num_frames, target_frames=300)

        seq = []
        for idx in indices:
            row = data[start + idx, :]
            hand_data = convert_row_to_shadow_angles(row, movement, rep, mat_filename)
            seq.append([getattr(hand_data, joint) for joint in joint_names])

        seq_array = np.array(seq, dtype=np.float32)
        sequences.append(seq_array)
        movement_ids.append(movement)

        if exercise_table == 'B':
            movement_name = MOVEMENT_NAMES_B.get(movement, "Unknown")
        else:
            movement_name = MOVEMENT_NAMES_C.get(movement, "Unknown")

        movement_names.append(movement_name)
        subject_ids.append(subject_id)
        exercise_tables.append(exercise_table)
        exercise_ids.append(exercise_id)
        session_ids.append(session_id)
        valid_lengths.append(seq_array.shape[0])

    # Pad sequences to the same length for transformer readiness
    max_len = max(valid_lengths) if sequences else 0
    num_joints = sequences[0].shape[1] if sequences else 0

    padded_sequences = np.zeros((len(sequences), max_len, num_joints), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.shape[0], :] = seq

    # Save as .npz
    np.savez_compressed(
        output_filename,
        sequences=padded_sequences,
        valid_lengths=np.array(valid_lengths),
        movement_ids=np.array(movement_ids),
        movement_names=np.array(movement_names),
        subject_ids=np.array(subject_ids),
        exercise_tables=np.array(exercise_tables),
        exercise_ids=np.array(exercise_ids),
        session_ids=np.array(session_ids)
    )

    print(f"✅ Dataset saved: {output_filename} | Total sequences: {len(sequences)} (max length: {max_len})")
