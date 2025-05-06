import sys
import socket
import threading
from sendAngles import send_from_mat, input_thread
from makeShadowDataset import generate_shadow_dataset
from makeFullShadowSet import export_full_dataset_to_hdf5

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python converter_main.py send <filename.mat>        # Send angles over UDP")
        print("  python converter_main.py export <filename.mat>      # Generate dataset from one file")
        print("  python converter_main.py export-all                 # Generate dataset from all files")
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
        output_filename = mat_filename.replace('.mat', '_shadow_dataset.npz')
        generate_shadow_dataset(mat_filename, output_filename)
    elif mode == "export-all":
        export_full_dataset_to_hdf5()
    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
