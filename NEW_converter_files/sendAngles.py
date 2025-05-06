import os
import time
import numpy as np
import threading
import scipy.io as sio
from shadow_utils import (
    scan_dataset_for_ranges,
    convert_row_to_shadow_angles,
    get_baseline_hand_data,
    set_shadow_rest_baseline,
    downsample_indices
)

IP_SERVER = "127.0.0.1"
PORT_SERVER = 20001
SERVER_ADDRESS_PORT = (IP_SERVER, PORT_SERVER)
READING_SOCKET_DELAY = 0.01
pause_event = threading.Event()
pause_event.set()

RESET_FRAMES = 100  # Baseline frame count

def debug_print_joint_data(hand_data, frame_number, movement=None, repetition=None):
    print(f"Frame {frame_number} => Movement: {movement}, Repetition: {repetition}")
    for joint, value in sorted(hand_data.__dict__.items()):
        print(f"  {joint}: {value:.2f}")



def send_from_mat(mat_filename, udp_socket):
    # 🔄 Determine baseline file first
    mat_dir = os.path.dirname(mat_filename) or '.'
    subject_prefix = mat_filename.split('_')[0]  # e.g., "S1"
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat") and f.startswith(subject_prefix)]


    has_e1 = any("_E1_" in f for f in mat_files)
    has_e2 = any("_E2_" in f for f in mat_files)
    has_e3 = any("_E3_" in f for f in mat_files)

    if has_e1 and has_e2:
        baseline_file = next((f for f in mat_files if "_E1_" in f), None)
        baseline_desc = "E1"
    elif has_e2:
        baseline_file = next((f for f in mat_files if "_E2_" in f), None)
        baseline_desc = "E2"
    else:
        raise FileNotFoundError(f"No E1 or E2 file found in {mat_dir} to compute baseline!")

    baseline_path = os.path.join(mat_dir, baseline_file)
    print(f"🔄 Computing baseline from: {baseline_desc} ({baseline_path})")

    e_data = sio.loadmat(baseline_path)
    if 'angles' not in e_data or 'restimulus' not in e_data:
        raise ValueError(f"Baseline file {baseline_path} missing 'angles' or 'restimulus'")

    e_angles = e_data['angles']
    e_movements = e_data['restimulus'].flatten()
    e_repetitions = e_data.get('re_repetition') or e_data.get('repetition')
    if e_repetitions is None:
        e_repetitions = np.zeros_like(e_movements)
    else:
        e_repetitions = e_repetitions.flatten()

    set_shadow_rest_baseline(e_angles, e_movements, e_repetitions)

    # 🔍 Load the actual file to process
    mat_data = sio.loadmat(mat_filename)
    if 'angles' not in mat_data or 'restimulus' not in mat_data:
        print(".mat file missing 'angles' or 'restimulus'")
        return

    data = mat_data['angles']
    movement_data = mat_data['restimulus'].flatten()
    repetition = mat_data.get('re_repetition') or mat_data.get('repetition')
    if repetition is None:
        repetition = np.zeros_like(movement_data)
    else:
        repetition = repetition.flatten()

    print("Scanning dataset for min/max per finger + joint...")
    scan_dataset_for_ranges(data)

    # 🚀 Send initial baseline hand data once
    hand_data = get_baseline_hand_data()
    hand_data.convertToInt()
    packet = hand_data.to_struct()
    udp_socket.sendto(packet, SERVER_ADDRESS_PORT)
    time.sleep(5)

    skip_movements_b = {4, 5, 8}

    i = 0
    while i < len(movement_data):
        pause_event.wait()
        movement = int(movement_data[i])
        rep = int(repetition[i]) if repetition is not None else 0

        if movement == 0 or rep == 0:
            i += 1
            continue  # Skip rest frames and invalid repetitions

        # Determine if current file is Table B (for skipping logic)
        if "_E1_" in mat_filename:
            is_table_b = True
        elif "_E2_" in mat_filename and has_e3:
            is_table_b = True
        else:
            is_table_b = False

        if is_table_b and movement in skip_movements_b:
            print(f"⏭️ Skipping movement {movement} (Table B skip)")
            while i < len(movement_data) and movement_data[i] == movement:
                i += 1
            continue

        # ▶️ Detect repetition boundaries
        start = i
        while i < len(movement_data) and movement_data[i] == movement:
            i += 1
        end = i

        # ✅ Downsample indices for this repetition
        ds_idx = downsample_indices(end - start, target_frames=300)

        # 🚨 Inject baseline frames to reset before each repetition
        print(f"➡️ New repetition detected (Movement {movement}, Rep {rep}), resetting...")
        baseline_hand_data = convert_row_to_shadow_angles(data[0, :], 0, rep, mat_filename)
        for _ in range(RESET_FRAMES):
            debug_print_joint_data(baseline_hand_data, start, 0, 0)
            baseline_hand_data.convertToInt()
            packet = baseline_hand_data.to_struct()
            udp_socket.sendto(packet, SERVER_ADDRESS_PORT)
            time.sleep(READING_SOCKET_DELAY)

        # 🎯 Now process downsampled frames in this repetition
        for idx in ds_idx:
            pause_event.wait()
            row = data[start + idx, :]
            hand_data = convert_row_to_shadow_angles(row, movement, rep, mat_filename)
            debug_print_joint_data(hand_data, start + idx, movement, rep)
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
