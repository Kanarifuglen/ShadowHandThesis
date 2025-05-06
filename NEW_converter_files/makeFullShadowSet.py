import os
import numpy as np
import h5py
import scipy.io as sio
from tqdm import tqdm
from multiprocessing import Process
from collections import defaultdict
from shadow_utils import (
    set_shadow_rest_baseline,
    convert_row_to_shadow_angles,
    scan_dataset_for_ranges,
    downsample_indices,
    MOVEMENT_NAMES_B,
    MOVEMENT_NAMES_C
)

def fill_dataset_worker(thread_id, subject_dirs, joint_names, output_path, movement_name_maps):
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

                    num_frames = end - start
                    indices = downsample_indices(num_frames, target_frames=300)

                    frames = []
                    for idx in indices:
                        row = angles[start + idx, :]
                        hand_data = convert_row_to_shadow_angles(row, movement_id, rep_id, mat_file)
                        frame_data = [getattr(hand_data, joint) for joint in joint_names]
                        frames.append(frame_data)

                    frames_np = np.array(frames, dtype=np.float32)
                    valid_length = frames_np.shape[0]
                    max_len = 300  # We know it's padded to 300 max

                    # Pad if needed
                    if valid_length < max_len:
                        padded = np.zeros((max_len, frames_np.shape[1]), dtype=np.float32)
                        padded[:valid_length, :] = frames_np
                    else:
                        padded = frames_np

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
                    movement_grp.create_dataset("angles", data=padded)
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


def export_full_dataset_to_hdf5(num_threads=4):
    """Parses through every subject directory, processes them in parallel threads, and merges into one dataset."""
    DATA_DIR = r"C:\Master\ShadowHandMotionPrediction-1\kinematics_dataset"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shadow_dataset.h5")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Find all subject directories
    subject_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d))]

    # Split directories evenly across threads
    chunks = [subject_dirs[i::num_threads] for i in range(num_threads)]

    joint_names = [
        'rh_WRJ1', 'rh_WRJ2',
        'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
        'rh_LFJ5'
    ] + [f'rh_{finger}J{idx}' for finger in ['FF', 'MF', 'RF', 'LF'] for idx in range(1, 5)]

  
    movement_name_maps = {
        'B': MOVEMENT_NAMES_B,
        'C': MOVEMENT_NAMES_C
    }

    tmp_paths = [f"tmp_thread_{i}.h5" for i in range(num_threads)]

    # Launch parallel processes
    processes = []
    for i in range(num_threads):
        p = Process(target=fill_dataset_worker, args=(
            i, chunks[i], joint_names, tmp_paths[i], movement_name_maps))
        processes.append(p)

    print("🚀 Starting export workers...")
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("✅ All export workers completed.")

    # Merge into final HDF5
    with h5py.File(output_path, "w") as final_h5:
        root_grp = final_h5.create_group("movements")
        for tmp_path in tmp_paths:
            print(f"🔗 Merging {tmp_path}...")
            with h5py.File(tmp_path, "r") as tmp_h5:
                for key in tmp_h5["movements"]:
                    tmp_h5.copy(f"movements/{key}", root_grp)
    print(f"✅ Final dataset saved to {output_path}")

    # Optional preview
    with h5py.File(output_path, "r") as h5f:
        print("🔎 Sample Movement Groups:")
        for i, key in enumerate(list(h5f["movements"].keys())[:9]):
            grp = h5f["movements"][key]
            print(f"  {key}: "
                f"movement_id={grp.attrs.get('movement_id')}, "
                f"subject_id={grp.attrs.get('subject_id')}, "
                f"exercise_id={grp.attrs.get('exercise_id')}, "
                f"exercise_table={grp.attrs.get('exercise_table')}, "
                f"session_id={grp.attrs.get('session_id')}, "
                f"movement_name={grp.attrs.get('movement_name').decode() if isinstance(grp.attrs.get('movement_name'), bytes) else grp.attrs.get('movement_name')}, "
                f"valid_length={grp.attrs.get('valid_length')}")

