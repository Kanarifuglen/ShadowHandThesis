import os
import h5py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from multiprocessing import Process
from mat_worker import fill_dataset_worker
from shadow_utils import (
    MOVEMENT_NAMES_B,
    MOVEMENT_NAMES_C
)

def export_full_dataset_to_hdf5(num_threads=4, target_frames=130):
    """Parses through every subject directory, processes them in parallel threads, and merges into one dataset."""
    DATA_DIR = r"C:\Master\ShadowHandThesis\datasets\kinematics_dataset"
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets'))
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "shadow_dataset.h5")
    tmp_paths = [os.path.join(output_dir, f"tmp_thread_{i}.h5") for i in range(num_threads)]


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
            i, chunks[i], joint_names, tmp_paths[i], movement_name_maps, target_frames))
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

