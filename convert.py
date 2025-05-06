import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import re
import sys
import socket
import time

import threading
import scipy.io as sio
import numpy as np
import h5py

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

    for i in range(data.shape[0]):
        pause_event.wait()
        row = data[i, :]
        movement = movement_data[i]
        rep = int(repetition[i]) if repetition is not None else 0

        if movement == 0:
            continue
        else:
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



def export_full_dataset_to_hdf5():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(script_dir, "..", "..", "..", "kinematics_dataset")
    output_path = os.path.join(script_dir, "full_shadow_dataset.h5")

    print(f"📂 Scanning dataset directory: {DATA_DIR}")
    mat_files = []
    for root, _, files in os.walk(DATA_DIR):
        for filename in files:
            if filename.endswith(".mat") and re.match(r"S\d+_E[123]_A1\.mat", filename):
                mat_files.append(os.path.join(root, filename))

    print(f"🔍 Found {len(mat_files)} .mat files")

    joint_names = [
        'rh_WRJ1', 'rh_WRJ2',
        'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
        'rh_LFJ5'
    ] + [f'rh_{finger}J{idx}' for finger in ['FF', 'MF', 'RF', 'LF'] for idx in range(1, 5)]

    # Create HDF5 file with expandable datasets
    with h5py.File(output_path, "w") as h5file:
        maxshape = (None, len(joint_names))
        data_ds = h5file.create_dataset("data", shape=(0, len(joint_names)), maxshape=maxshape, dtype='f4', chunks=True)
        label_ds = h5file.create_dataset("movement_labels", shape=(0,), maxshape=(None,), dtype='i4', chunks=True)
        h5file.create_dataset("joint_names", data=np.array(joint_names, dtype='S'))

        total_frames = 0

        for idx, path in enumerate(mat_files, 1):
            print(f"[{idx}/{len(mat_files)}] Processing {os.path.basename(path)}")
            result = process_mat_file(path)
            if result is None:
                print(f"⚠️ Skipped {os.path.basename(path)}")
                continue

            frames, labels = result
            n_new = frames.shape[0]

            # Resize HDF5 datasets
            data_ds.resize((total_frames + n_new, len(joint_names)))
            label_ds.resize((total_frames + n_new,))
            data_ds[total_frames:total_frames + n_new, :] = frames
            label_ds[total_frames:total_frames + n_new] = labels

            total_frames += n_new

    print(f"✅ Done! Final dataset saved to {output_path}")
    print(f"📊 Total frames: {total_frames}")


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
