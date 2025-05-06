import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import re
import sys
import socket
import time
import h5py
import threading
import scipy.io as sio
import numpy as np
import h5py 
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
from multiprocessing import Process


from shadow_utils import (
    build_dynamic_ranges,
    convert_row_to_shadow_angles,
    NINAPRO_MAPPING,
    SHADOW_REST
)
from mat_worker import process_mat_file
from tools.handAngleData import HandAnglesData

# UDP Settings
IP_SERVER = "127.0.0.1"
PORT_SERVER = 20001
SERVER_ADDRESS_PORT = (IP_SERVER, PORT_SERVER)
READING_SOCKET_DELAY = 0.01
pause_event = threading.Event()
pause_event.set()

def debug_print_joint_data(hand_data, frame_number, movement=None, repetition=None):
    """Debug function to print out joint data per frame"""

    print(f"Frame {frame_number} => Movement: {movement}, Repetition: {repetition}")
    for joint, value in sorted(hand_data.__dict__.items()):
        print(f"  {joint}: {value:.2f}")

def send_from_mat(mat_filename, udp_socket):
    mat_data = sio.loadmat(mat_filename)
    if 'angles' not in mat_data or 'restimulus' not in mat_data:
        print(".mat file missing 'angles' or 'restimulus'")
        return

    data = mat_data['angles']
    movement_data = mat_data['restimulus'].flatten()
    repetition = mat_data.get('re_repetition') or mat_data.get('repetition')

    current_ranges = build_dynamic_ranges(data, movement_data)

    MAX_STATIC_FRAMES = 300
    last_movement = None
    static_frame_counter = 0

    for i in range(data.shape[0]):
        pause_event.wait()
        row = data[i, :]
        movement = movement_data[i]
        rep = int(repetition[i]) if repetition is not None else 0

        if movement == 0:
            continue

        if movement != last_movement:
            static_frame_counter = 0
            last_movement = movement
        else:
            static_frame_counter += 1
            if static_frame_counter > MAX_STATIC_FRAMES:
                continue  # Discard frame if movement is static too long

        hand_data = convert_row_to_shadow_angles(row, movement, current_ranges, mat_filename)
        debug_print_joint_data(hand_data, i, movement, rep)
        hand_data.convertToInt()
        packet = hand_data.to_struct()
        udp_socket.sendto(packet, SERVER_ADDRESS_PORT)
        time.sleep(READING_SOCKET_DELAY)


def input_thread():
    while True:
        cmd = input("[p=Pause, r=Resume, q=Quit]: ").strip().lower()
        if cmd == 'p': pause_event.clear()
        elif cmd == 'r': pause_event.set()
        elif cmd == 'q': sys.exit(0)



def generate_shadow_dataset(mat_filename, output_filename="shadow_dataset.npz"):
    """Generates a dataset from a single .mat file"""

    mat_data = sio.loadmat(mat_filename)
    if 'angles' not in mat_data or 'restimulus' not in mat_data:
        raise ValueError("Missing 'angles' or 'restimulus' in .mat file.")

    data = mat_data['angles']
    movement_data = mat_data['restimulus'].flatten()
    current_ranges = build_dynamic_ranges(data, movement_data)

    all_frames = []
    all_movements = []

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
            if rest_frames_added >= 5:
                continue
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
            hand_data = convert_row_to_shadow_angles(row, movement, current_ranges, mat_filename)

        all_frames.append([getattr(hand_data, joint) for joint in joint_names])
        all_movements.append(movement)

    np_array = np.array(all_frames)
    np_movements = np.array(all_movements)

    np.savez_compressed(output_filename, data=np_array, joint_names=joint_names, movement_labels=np_movements)
    print(f"✅ Dataset saved: {output_filename} | Total frames: {np_array.shape[0]}")


def fill_names(movement_items):
    movement_names = ["" for _ in range(len(movement_items))]
    for k, v in movement_items:
        movement_names[k] = v
    movement_names = movement_names[:-1]
    return movement_names


def fill_dataset_worker(m_id, mat_files, joint_names, movement_names_dict, output_path):
    """Thread which job is to fill out a hp5y file for exercise A, B or C"""

    movement_instance_counts = defaultdict(int)

    with h5py.File(output_path, "w") as h5f:
        root_grp = h5f.create_group("movements")

        position = 0 if m_id == 'A' else 1 if m_id == 'B' else 2
        progress_bar = tqdm(mat_files, desc=f"[{m_id}] Processing files", position=position)

        for full_path in progress_bar:
            filename = os.path.basename(full_path)
            session_id = os.path.splitext(filename)[0]
            subject_id = session_id.split("_")[0]
            exercise_id = session_id.split("_")[1]

            try:
                mat = sio.loadmat(full_path)
                if 'angles' not in mat or 'restimulus' not in mat:
                    continue

                angles = mat['angles']
                labels = mat['restimulus'].flatten()
                current_ranges = build_dynamic_ranges(angles, labels)

                j = 0
                while j < len(labels):
                    movement_id = labels[j]
                    if movement_id == 0:
                        j += 1
                        continue
                    start = j
                    while j < len(labels) and labels[j] == movement_id:
                        j += 1
                    end = j

                    frames = []
                    for k in range(start, end):
                        row = angles[k, :]
                        hand_data = convert_row_to_shadow_angles(row, movement_id, current_ranges, full_path)
                        frame_data = [getattr(hand_data, joint) for joint in joint_names]
                        frames.append(frame_data)

                    frames_np = np.array(frames, dtype=np.float32)

                    key_base = f"{exercise_id}__movement_{movement_id:03d}__{subject_id}"
                    movement_instance_counts[key_base] += 1
                    instance_num = movement_instance_counts[key_base]
                    group_name = f"{key_base}_{instance_num:02d}"

                    movement_grp = root_grp.create_group(group_name)
                    movement_grp.create_dataset("angles", data=frames_np)
                    movement_grp.attrs["movement_id"] = int(movement_id)
                    movement_grp.attrs["movement_name"] = movement_names_dict.get(int(movement_id), "Unknown")
                    movement_grp.attrs["session"] = session_id

            except Exception:
                continue

        #print summary once at end
        total = sum(movement_instance_counts.values())
        tqdm.write(f"[{m_id}] ✅ Done: {total} movement segments written.")



def export_full_dataset_to_hdf5():
    """Method which parses through every .mat file in the ninapro dataset and converts them to one shadow hand angles dataset. Each thread handles their own set of exercises, before combing into 
    one final dataset at the end"""

    DATA_DIR = "/mnt/c/Master/ShadowHandMotionPrediction-1/kinematics_dataset"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exercise_shadow_dataset.h5")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    joint_names = [
        'rh_WRJ1', 'rh_WRJ2',
        'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
        'rh_LFJ5'
    ] + [f'rh_{finger}J{idx}' for finger in ['FF', 'MF', 'RF', 'LF'] for idx in range(1, 5)]

    MOVEMENT_NAMES_A = {
        1: "Index flexion", 2: "Index extension", 3: "Middle flexion", 4: "Middle extension",
        5: "Ring flexion", 6: "Ring extension", 7: "Little finger flexion", 8: "Little finger extension", 9: "Thumb adduction",
        10: "Thumb abduction", 11 : "Thumb flexion", 12 : "Thumb extension",   0: "Rest"
    }

    MOVEMENT_NAMES_B = {
        1: "Thumb up", 2: "Extension of index and middle, flexion of the others", 3: "Flexion of ring and little finger, extension of the others",
        4: "Thumb opposing base of little finger", 5: "Abduction of all fingers", 6: "Fingers flexed together into a fist", 7: "Pointing index", 8: "Adduction on extended fingers", 9: "Wrist supination (MF axis)",
        10: "Wrist supination", 11 : "Wrist supination (LF axis)", 12 : "Wrist pronation (LF axis)", 13 : "Wrist flexion", 
        14: "Wrist extension", 15: "Wrist radial devation", 16 : "Wrist ulnar deviation", 17: "Wrist extension with closed hand", 0: "Rest"
    }

    MOVEMENT_NAMES_C = {
        1: "Large diameter grasp", 2: "Small diameter grasp", 3: "Fixed hook grasp", 4: "Index finger extension grasp",
        5: "Medium wrap", 6: "Ring grasp", 7: "Prismatic four fingers grasp", 8: "Stick grasp", 9: "Writing tripod grasp",
        10: "Power sphere grasp", 11 : "Three finger sphere grasp", 12 : "Precision sphere grasp", 13: "Tripod grasp", 14: "Prismatic pinch grasp", 
        15: "Tip pinch grasp", 16: "Quadpod grasp", 17: "Lateral grasp", 18: "Parallell extension grasp", 19 : "Extension type grasp", 20: "Power disk grasp",
        21: "Open a bottle with a tripod grasp", 22: "Turn a screw", 23: "Cut something",  0: "Rest"
    }

    mat_files_A = []
    mat_files_B = []
    mat_files_C = []
    for root, _, files in os.walk(DATA_DIR):
        for filename in files:
            if filename.endswith(".mat") and re.match(r"S\d+_E[1]_A1\.mat", filename):
                mat_files_A.append(os.path.join(root, filename))
            if filename.endswith(".mat") and re.match(r"S\d+_E[2]_A1\.mat", filename):
                mat_files_B.append(os.path.join(root, filename))
            if filename.endswith(".mat") and re.match(r"S\d+_E[3]_A1\.mat", filename):
                mat_files_C.append(os.path.join(root, filename))

    print("Len A " , len(mat_files_A))
    print("Len B " , len(mat_files_B))
    print("Len C " , len(mat_files_C))
    tmp_paths = {
        'A': "tmp_A.h5",
        'B': "tmp_B.h5",
        'C': "tmp_C.h5"
    }

    # Launch parallel processes
    processes = [
        Process(target=fill_dataset_worker, args=('A', mat_files_A, joint_names, MOVEMENT_NAMES_A, tmp_paths['A'])),
        Process(target=fill_dataset_worker, args=('B', mat_files_B, joint_names, MOVEMENT_NAMES_B, tmp_paths['B'])),
        Process(target=fill_dataset_worker, args=('C', mat_files_C, joint_names, MOVEMENT_NAMES_C, tmp_paths['C']))
    ]

    print("🚀 Starting export workers...")
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("✅ All export workers completed.")

    # Merge into final HDF5
    with h5py.File(output_path, "w") as final_h5:
        root_grp = final_h5.create_group("movements")
        for tag, tmp_path in tmp_paths.items():
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
                  f"session={grp.attrs.get('session')}, "
                  f"movement_name={grp.attrs.get('movement_name').decode() if isinstance(grp.attrs.get('movement_name'), bytes) else grp.attrs.get('movement_name')}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python converter.py send <filename.mat>        # Send angles over UDP")
        print("  python converter.py export <filename.mat>      # Generate dataset from one file")
        print("  python converter.py export-all                 # Generate dataset from all files")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "send":
        mat_filename = sys.argv[2]
        if not os.path.isfile(mat_filename):
            mat_filename = os.path.join("kinematics_send", mat_filename)
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        threading.Thread(target=input_thread, daemon=True).start()
        send_from_mat(mat_filename, udp_socket)
        print("✅ Finished sending frames.")
    elif mode == "export":
        mat_filename = sys.argv[2]
        output_filename = os.path.splitext(mat_filename)[0] + "_shadow_dataset.npz"
        generate_shadow_dataset(mat_filename, output_filename)
    elif mode == "export-all":
        export_full_dataset_to_hdf5()
    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
