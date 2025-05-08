import sys
import socket
import threading
from sendAngles import send_from_mat, input_thread
from makeShadowDataset import generate_shadow_dataset
from makeFullShadowSet import export_full_dataset_to_hdf5

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python converter_main.py send <filename.mat> [target_frames]")
        print("  python converter_main.py export <filename.mat> [target_frames]")
        print("  python converter_main.py export-all [target_frames]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    target_frames = 130  # default value

    # Optional target_frames argument
    if len(sys.argv) >= 4 and sys.argv[3].isdigit():
        target_frames = int(sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[2].isdigit():
        target_frames = int(sys.argv[2])

    if mode == "send":
        mat_filename = sys.argv[2]
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        threading.Thread(target=input_thread, daemon=True).start()
        send_from_mat(mat_filename, udp_socket, target_frames=target_frames)
        print("✅ Finished sending frames.")
    elif mode == "export":
        mat_filename = sys.argv[2]
        output_filename = mat_filename.replace('.mat', f'_shadow_dataset_{target_frames}.npz')
        generate_shadow_dataset(mat_filename, output_filename, target_frames=target_frames)
    elif mode == "export-all":
        export_full_dataset_to_hdf5(target_frames=target_frames)
    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
