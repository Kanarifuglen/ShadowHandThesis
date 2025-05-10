"""
This script is based on and inspired by 'converterRokokoShadowHand.py',
developed by Charlotte Andreff and Loïc Blanc, SINLAB, University of Oslo (2024).
Repository: https://github.com/sinlab-uio/rokoko-to-robots
"""

import socket
import time
import os
import json
import sys
import threading
import numpy as np
import argparse
import h5py
from tqdm import tqdm
from collections import defaultdict
from shadow_utils import resample_sequence



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.handAngleData import HandAnglesData
from shadow_utils import resample_sequence  # Import the resampling function

IP_SERVER = "127.0.0.1"
PORT_SERVER = 20001
BUFFER_SIZE = 16384
READING_SOCKET_DELAY = 0.1
SERVER_ADDRESS_PORT = (IP_SERVER, PORT_SERVER)

# Events:
# - pause_event:  set => playing, clear => paused
# - skip_file_event: set => skip current file
# - skip_dir_event:  set => skip current directory
pause_event = threading.Event()
skip_file_event = threading.Event()
skip_dir_event = threading.Event()

script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    script_dir, 
    "..", "..", "..",
    "SMOKS_Dataset"
)

def convertFromJSONDataToAnglesData(angles_dict):
    return HandAnglesData.fromDict(angles_dict)

def resample_json_frames(frames, target_frames=130):
    """
    Resample a list of JSON frames to target_frames using linear interpolation.
    """
    if len(frames) == target_frames:
        return frames  # Already correct length
    
    # Extract features from frames as a numpy array
    keys = frames[0].keys()
    feature_array = np.zeros((len(frames), len(keys)))
    
    # Convert frames to a 2D array (frames × features)
    for i, frame in enumerate(frames):
        for j, key in enumerate(keys):
            feature_array[i, j] = frame[key]
    
    # Resample using the utility function
    resampled_array = resample_sequence(feature_array, target_frames)
    
    # Convert back to list of dictionaries
    resampled_frames = []
    for i in range(target_frames):
        frame_dict = {}
        for j, key in enumerate(keys):
            frame_dict[key] = float(resampled_array[i, j])
        resampled_frames.append(frame_dict)
    
    return resampled_frames

def sendAngles(frames, udp_socket, target_frames=130):
    """
    Normalizes frames to target_frames and sends them over UDP,
    respecting pause_event and skip_file_event.
    Returns True if we finished normally, or False if we had to skip.
    """
    # Normalize frames to target length
    normalized_frames = resample_json_frames(frames, target_frames)
    print(f"➡️ Resampled sequence from {len(frames)} → {len(normalized_frames)} frames")
    
    for frame in normalized_frames:
        # Check if we need to skip the rest of this file
        if skip_file_event.is_set() or skip_dir_event.is_set():
            # Clear skip_file_event so it doesn't affect the next file
            skip_file_event.clear()
            return False  # Indicate we didn't finish normally

        # Wait here if paused
        pause_event.wait()

        # Build data
        angles = convertFromJSONDataToAnglesData(frame)
        angles.convertToInt()
        data_to_send = angles.to_struct()

        # Send
        udp_socket.sendto(data_to_send, SERVER_ADDRESS_PORT)
        time.sleep(READING_SOCKET_DELAY)

    # If we get here, we finished all frames with no skip request
    return True

def input_thread():
    """
    Wait for user commands on the console:
      p = Pause
      r = Resume
      n = skip rest of current file
      d = skip entire directory
      q = Quit
    """
    while True:
        cmd = input("[p=Pause, r=Resume, n=Next file, d=Next directory, q=Quit]: ").strip().lower()
        if cmd == 'p':
            print("Pausing...")
            pause_event.clear()
        elif cmd == 'r':
            print("Resuming...")
            pause_event.set()
        elif cmd == 'n':
            print("Skipping current file...")
            skip_file_event.set()
        elif cmd == 'd':
            print("Skipping entire directory...")
            skip_dir_event.set()
        elif cmd == 'q':
            print("Quitting...")
            sys.exit(0)
        else:
            print("Unknown command. Use p, r, n, d, or q.")


def export_jsons_to_hdf5(json_root_dir, output_h5_path, target_frames=130):
    joint_names = [
        'rh_WRJ1', 'rh_WRJ2',
        'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
        'rh_LFJ5'
    ] + [f'rh_{finger}J{idx}' for finger in ['FF', 'MF', 'RF', 'LF'] for idx in range(1, 5)]

    with h5py.File(output_h5_path, "w") as h5f:
        root_grp = h5f.create_group("movements")

        subjects = [d for d in os.listdir(json_root_dir) if os.path.isdir(os.path.join(json_root_dir, d))]

        for subject_id in tqdm(subjects, desc="Subjects"):
            subject_path = os.path.join(json_root_dir, subject_id)
            poses = [p for p in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, p))]

            for pose_idx, pose_name in enumerate(poses, start=1):
                pose_path = os.path.join(subject_path, pose_name)
                json_files = sorted([f for f in os.listdir(pose_path) if f.endswith(".json")])

                # ✅ movement_id same for ALL subjects+reps in this pose
                movement_id = pose_idx

                for rep_idx, json_file in enumerate(json_files, start=1):
                    json_path = os.path.join(pose_path, json_file)
                    try:
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)

                        sequence_rows = []
                        for frame_dict in json_data:
                            hand_angles = HandAnglesData.fromDict(frame_dict)
                            row = [getattr(hand_angles, joint) for joint in joint_names]
                            sequence_rows.append(row)

                        sequence = np.array(sequence_rows, dtype=np.float32)

                        if sequence.ndim != 2 or sequence.shape[1] != 24:
                            print(f"⚠️ Skipping {json_path}: unexpected shape {sequence.shape}")
                            continue

                        resampled_sequence = resample_sequence(sequence, target_frames)

                        # ✅ unique group key = pose + subject + repetition
                        key_base = f"{pose_name}__movement_{movement_id:03d}__{subject_id}__rep_{rep_idx}"
                        movement_group = root_grp.create_group(key_base)
                        movement_group.create_dataset("angles", data=resampled_sequence)

                        # ✅ movement_id = pose index
                        movement_group.attrs["movement_id"] = movement_id
                        movement_group.attrs["movement_name"] = pose_name
                        movement_group.attrs["session_id"] = f"{subject_id}_{pose_name}"
                        movement_group.attrs["subject_id"] = subject_id
                        movement_group.attrs["exercise_id"] = pose_name
                        movement_group.attrs["exercise_table"] = "JSON"
                        movement_group.attrs["valid_length"] = target_frames
                        movement_group.attrs["repetition_id"] = rep_idx

                        print(f"✅ Saved {key_base}")

                    except Exception as e:
                        print(f"❌ Error processing {json_path}: {e}")

    print(f"✅ Finished export: {output_h5_path}")





def preview_hdf5_dataset(h5_path):
    with h5py.File(h5_path, "r") as h5f:
        movements_grp = h5f["movements"]
        keys = list(movements_grp.keys())

        print(f"✅ Dataset contains {len(keys)} movement entries.")

        # Group keys by subject + pose
        subject_pose_groups = {}
        for key in keys:
            parts = key.split("__")
            pose_name = parts[0]
            subject_id = parts[2]
            group_key = (subject_id, pose_name)
            if group_key not in subject_pose_groups:
                subject_pose_groups[group_key] = []
            subject_pose_groups[group_key].append(key)

        for (subject_id, pose_name), group_keys in subject_pose_groups.items():
            print(f"\n--- Subject: {subject_id}, Pose: {pose_name} ---")
            for idx, key in enumerate(sorted(group_keys)[:2]):  # ✅ print up to 2 reps
                grp = movements_grp[key]
                print(f"\n[{idx+1}] Movement group: {key}")
                print(f"  movement_id: {grp.attrs.get('movement_id')}")
                print(f"  movement_name: {grp.attrs.get('movement_name')}")
                print(f"  session_id: {grp.attrs.get('session_id')}")
                print(f"  subject_id: {grp.attrs.get('subject_id')}")
                print(f"  exercise_id: {grp.attrs.get('exercise_id')}")
                print(f"  exercise_table: {grp.attrs.get('exercise_table')}")
                print(f"  repetition_id: {grp.attrs.get('repetition_id')}")
                print(f"  valid_length: {grp.attrs.get('valid_length')}")
                print(f"  angles shape: {grp['angles'].shape}")





def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Send JSON hand animation frames over UDP with normalization")
    parser.add_argument("--frames", type=int, default=130, help="Target number of frames to normalize to (default: 130)")
    parser.add_argument('--data-dir', type=str, default=r"C:\Master\ShadowHandThesis\datasets\SMOKS_Dataset", help="Root directory of JSON dataset")
    parser.add_argument("--export-hdf5", action="store_true", help="Export JSONs to HDF5 instead of sending UDP")
    parser.add_argument("--output-hdf5", type=str, default="../datasets/json_dataset.h5")
    parser.add_argument("--test", action="store_true", help="Preview contents of the HDF5 dataset")

    args = parser.parse_args()
    
    if args.test:
        preview_hdf5_dataset(args.output_hdf5)
        return
    if args.export_hdf5:
        export_jsons_to_hdf5(args.data_dir, args.output_hdf5, target_frames=args.frames)
    else:
        # Create UDP socket
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Start input thread so user can type commands
        t = threading.Thread(target=input_thread, daemon=True)
        t.start()

        # Initially we are "playing" (not paused)
        pause_event.set()

        data_dir = args.data_dir
        
        print(f"Converting and sending JSON files with {args.frames} normalized frames per file")
        
        for root, dirs, files in os.walk(data_dir):
            # Skip SMOKS_Dataset itself
            if root == data_dir:
                continue

            current_dir_name = os.path.basename(root)
            parent_dir_name = os.path.basename(os.path.dirname(root))

            # Print out directory name if it's top-level or starts with 'pose'
            if parent_dir_name == os.path.basename(data_dir):
                print(f"Now entering directory: {current_dir_name}")
            elif current_dir_name.startswith("pose"):
                print(f"Now entering directory: {current_dir_name}")

            files.sort()
            skip_current_directory = False

            for file_name in files:
                if file_name.endswith('.json'):
                    if skip_dir_event.is_set():
                        # The user typed 'd' to skip entire directory
                        skip_dir_event.clear()  # Reset for next directory
                        skip_current_directory = True
                        break

                    print(f"  Processing animation for {file_name}")
                    file_path = os.path.join(root, file_name)
                    
                    try:
                        with open(file_path, "r") as f:
                            frames = json.load(f)
                        
                        if len(frames) < 10:
                            print(f"⚠️ Skipping {file_name}: too few frames ({len(frames)})")
                            continue
                            
                        print(f"  Original frames: {len(frames)}")
                        finished_normally = sendAngles(frames, udp_socket, target_frames=args.frames)
                        
                        if not finished_normally:
                            # We must have triggered skip_file_event
                            print(f"  Skipped {file_name}\n")
                        else:
                            print(f"  ✅ Successfully sent normalized {file_name}\n")
                    
                    except Exception as e:
                        print(f"  ❌ Error processing {file_name}: {e}\n")

                    time.sleep(2.0)

            if skip_current_directory:
                print(f"Skipping remainder of directory {current_dir_name}")
                continue

        print("All directories processed. Exiting.")

if __name__ == "__main__":
    main()