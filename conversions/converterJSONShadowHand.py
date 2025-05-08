import socket
import time
import os
import json
import sys
import threading


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.handAngleData import HandAnglesData

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

def sendAngles(frames, udp_socket):
    """
    Sends all frames in 'frames', but respects pause_event and skip_file_event.
    Returns True if we finished normally, or False if we had to skip.
    """
    for frame in frames:
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

def main():
    # Create UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Start input thread so user can type commands
    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    # Initially we are "playing" (not paused)
    pause_event.set()

    for root, dirs, files in os.walk(DATA_DIR):
        # Remove 'male_noobject' directory if it exists
        if "male_noobject" in dirs:
            dirs.remove("male_noobject")

        # Skip SMOKS_Dataset itself
        if root == DATA_DIR:
            continue

        current_dir_name = os.path.basename(root)
        parent_dir_name = os.path.basename(os.path.dirname(root))

        # Print out directory name if it's top-level or starts with 'pose'
        if parent_dir_name == os.path.basename(DATA_DIR):
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

                print(f"  Playing animation for {file_name}")
                file_path = os.path.join(root, file_name)
                with open(file_path, "r") as f:
                    frames = json.load(f)

                finished_normally = sendAngles(frames, udp_socket)
                if not finished_normally:
                    # We must have triggered skip_file_event
                    print(f"  Skipped {file_name}\n")
                else:
                    print(f"  Done playing {file_name}\n")

                time.sleep(2.0)

        if skip_current_directory:
            print(f"Skipping remainder of directory {current_dir_name}")
            continue

    print("All directories processed. Exiting.")

if __name__ == "__main__":
    main()
