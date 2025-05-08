import os
import numpy as np
import h5py
import scipy.io as sio
from tqdm import tqdm
from collections import defaultdict
from shadow_utils import (
    set_shadow_rest_baseline,
    convert_row_to_shadow_angles,
    scan_dataset_for_ranges,
    resample_sequence,
)


def fill_dataset_worker(thread_id, subject_dirs, joint_names, output_path, movement_name_maps, target_frames):
    """Thread job: process all .mat files in its assigned directories."""
    movement_instance_counts = defaultdict(int)

    with h5py.File(output_path, "w") as h5f:
        root_grp = h5f.create_group("movements")

        progress_bar = tqdm(subject_dirs, desc=f"[Thread {thread_id}] Processing dirs", position=thread_id)
        for subject_dir in progress_bar:
            mat_files = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith(".mat")]

            has_e1 = any("_E1_" in f for f in mat_files)
            has_e2 = any("_E2_" in f for f in mat_files)

            # Find baseline file
            if has_e1 and has_e2:
                baseline_file = next((f for f in mat_files if "_E1_" in f), None)
                baseline_desc = "E1 (special case)"
            elif has_e2:
                baseline_file = next((f for f in mat_files if "_E2_" in f), None)
                baseline_desc = "E2"
            else:
                print(f"⚠️ Skipping {subject_dir}: No E1 or E2 file found for baseline.")
                continue

            #tqdm.write(f"🔄 [Thread {thread_id}] Computing baseline from: {baseline_desc} ({baseline_file})")


            baseline_data = sio.loadmat(baseline_file)
            if 'angles' not in baseline_data or 'restimulus' not in baseline_data:
                print(f"⚠️ Skipping {subject_dir}: Baseline file missing 'angles' or 'restimulus'")
                continue

            angles = baseline_data['angles']
            movements = baseline_data['restimulus'].flatten()
            repetitions = baseline_data.get('re_repetition') or baseline_data.get('repetition')
            if repetitions is None:
                repetitions = np.zeros_like(movements)
            else:
                repetitions = repetitions.flatten()
            set_shadow_rest_baseline(angles, movements, repetitions)

            # Process all mat files in this subject dir
            for mat_file in mat_files:
                filename = os.path.basename(mat_file)
                session_id = os.path.splitext(filename)[0]
                subject_id = session_id.split("_")[0]
                exercise_id = session_id.split("_")[1]

                mat = sio.loadmat(mat_file)
                if 'angles' not in mat or 'restimulus' not in mat:
                    continue

                angles = mat['angles']
                scan_dataset_for_ranges(angles)
                labels = mat['restimulus'].flatten()
                repetitions = mat.get('re_repetition') or mat.get('repetition')
                if repetitions is None:
                    repetitions = np.zeros_like(labels)
                else:
                    repetitions = repetitions.flatten()

                # Determine if this file is Table B
                mat_files_in_dir = [f for f in os.listdir(subject_dir) if f.endswith(".mat")]
                has_e3 = any("_E3_" in f for f in mat_files_in_dir)
                is_table_b = False
                if "_E1_" in mat_file:
                    is_table_b = True  # E1 always Table B
                elif "_E2_" in mat_file and has_e3:
                    is_table_b = True  # E2 + E3 dir → E2 is Table B
                else:
                    is_table_b = False  # Table C

                skip_movements_b = {4, 5, 8}
                exercise_table = 'B' if is_table_b else 'C'

                j = 0
                while j < len(labels):
                    movement_id = int(labels[j])
                    rep_id = int(repetitions[j]) if repetitions is not None else 0

                    if movement_id == 0 or rep_id == 0:
                        j += 1
                        continue

                    if is_table_b and movement_id in skip_movements_b:
                       # print(f"⏭️ Skipping movement {movement_id} (Table B skip) in {mat_file}")
                        while j < len(labels) and labels[j] == movement_id:
                            j += 1
                        continue

                    start = j
                    while j < len(labels) and labels[j] == movement_id and repetitions[j] == rep_id:
                        j += 1
                    end = j


                    frames_raw = []
                    for idx in range(start, end):
                        row = angles[idx, :]
                        hand_data = convert_row_to_shadow_angles(row, movement_id, rep_id, mat_file)
                        frame_data = [getattr(hand_data, joint) for joint in joint_names]
                        frames_raw.append(frame_data)

                    frames_raw_np = np.array(frames_raw, dtype=np.float32)

                    # ✅ Resample directly to 130 frames
                    frames_np = resample_sequence(frames_raw_np, target_frames)
                    valid_length = frames_np.shape[0]
                  
                    # Select correct movement map
                    if "_E1_" in mat_file:
                        movement_map = movement_name_maps['B']  # Table A handled as B here
                    elif "_E2_" in mat_file:
                        movement_map = movement_name_maps['B']
                    elif "_E3_" in mat_file:
                        movement_map = movement_name_maps['C']
                    else:
                        movement_map = {}

                    key_base = f"{exercise_id}__movement_{movement_id:03d}__{subject_id}"
                    movement_instance_counts[key_base] += 1
                    instance_num = movement_instance_counts[key_base]
                    group_name = f"{key_base}_{instance_num:02d}"

                    movement_grp = root_grp.create_group(group_name)
                    movement_grp.create_dataset("angles", data=frames_np)
                    movement_grp.attrs["movement_id"] = int(movement_id)
                    movement_grp.attrs["movement_name"] = movement_map.get(int(movement_id), "Unknown")
                    movement_grp.attrs["session_id"] = session_id
                    movement_grp.attrs["subject_id"] = subject_id
                    movement_grp.attrs["exercise_id"] = exercise_id
                    movement_grp.attrs["exercise_table"] = exercise_table
                    movement_grp.attrs["valid_length"] = valid_length
                    movement_grp.attrs["repetition_id"] = rep_id


        total = sum(movement_instance_counts.values())
        tqdm.write(f"[Thread {thread_id}] ✅ Done: {total} movement segments written.")
