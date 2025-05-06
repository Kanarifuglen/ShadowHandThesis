import os
import h5py
import numpy as np
from tqdm import tqdm

T_OBS = 30
T_PRED = 40
T_TOTAL = T_OBS + T_PRED
STATIC_FRAME_THRESHOLD = 0.8
STATIC_DIFF_THRESHOLD = 1e-3


def is_mostly_static(angles):
    frame_diff = np.abs(np.diff(angles, axis=0))
    static_frames = (frame_diff < STATIC_DIFF_THRESHOLD).all(axis=1)
    static_ratio = static_frames.sum() / static_frames.shape[0]
    return static_ratio > STATIC_FRAME_THRESHOLD


def create_filtered_h5(input_path, output_path):
    print(f"🔍 Loading original HDF5: {input_path}")
    h5_in = h5py.File(input_path, 'r')
    h5_out = h5py.File(output_path, 'w')

    source_group = h5_in['movements']
    target_group = h5_out.create_group('movements')

    total_added = 0
    total_skipped = 0

    for key in tqdm(source_group.keys(), desc="Filtering segments"):
        angles = source_group[key]['angles'][:].astype(np.float32)
        if angles.shape[0] < T_TOTAL:
            total_skipped += 1
            continue

        for t in range(0, angles.shape[0] - T_TOTAL):
            obs = angles[t:t + T_OBS]
            fut = angles[t + T_OBS:t + T_TOTAL]
            full = np.concatenate([obs, fut], axis=0)

            if is_mostly_static(full):
                total_skipped += 1
                continue

            new_key = f"{key}_{t}"
            g = target_group.create_group(new_key)
            g.create_dataset('angles', data=full)
            total_added += 1

    h5_in.close()
    h5_out.close()

    print(f"✅ Done. Added {total_added} segments. Skipped {total_skipped} static segments.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to original HDF5 file")
    parser.add_argument('--output', type=str, default="filtered_dataset.h5", help="Path to save filtered HDF5")
    args = parser.parse_args()

    create_filtered_h5(args.input, args.output)


    FF | MCP: raw=1.72, rel=-22.28, norm=0.47, scaled=34.81 | PIP: raw=-0.37, rel=-32.37, norm=0.21, scaled=18.48 | DIP: raw=-4.64, rel=-7.64, norm=0.07, scaled=7.46
MF | MCP: raw=-11.06, rel=-26.06, norm=0.20, scaled=7.00 | PIP: raw=3.23, rel=-16.77, norm=0.49, scaled=26.51 | DIP: raw=-9.93, rel=-19.93, norm=0.78, scaled=84.71
RF | MCP: raw=-4.28, rel=-16.28, norm=0.25, scaled=9.56 | PIP: raw=15.09, rel=-4.91, norm=0.51, scaled=41.12 | DIP: raw=-29.24, rel=-39.24, norm=0.83, scaled=74.54
LF | MCP: raw=3.51, rel=1.51, norm=0.21, scaled=4.96 | PIP: raw=-3.73, rel=-3.73, norm=0.05, scaled=3.21 | DIP: raw=-3.93, rel=-48.93, norm=0.56, scaled=40.41
Frame 15826 => Movement: 4, Repetition: 1
  rh_FFJ1: 7.46
  rh_FFJ2: 18.48
  rh_FFJ3: 34.81
  rh_FFJ4: 0.00
  rh_LFJ1: 40.41
  rh_LFJ2: 3.21
  rh_LFJ3: 4.96
  rh_LFJ4: 0.00
  rh_LFJ5: -1.02
  rh_MFJ1: 84.71
  rh_MFJ2: 26.51
  rh_MFJ3: 7.00
  rh_MFJ4: 0.00
  rh_RFJ1: 74.54
  rh_RFJ2: 41.12
  rh_RFJ3: 9.56
  rh_RFJ4: 0.00
  rh_THJ1: 0.00
  rh_THJ2: 0.00
  rh_THJ3: 0.00
  rh_THJ4: 0.00
  rh_THJ5: 0.00
  rh_WRJ1: -7.85
  rh_WRJ2: 0.92
FF | MCP: raw=2.21, rel=-21.79, norm=0.48, scaled=35.26 | PIP: raw=-0.12, rel=-32.12, norm=0.21, scaled=18.63 | DIP: raw=-4.93, rel=-7.93, norm=0.07, scaled=7.30
MF | MCP: raw=-11.06, rel=-26.06, norm=0.20, scaled=7.00 | PIP: raw=2.95, rel=-17.05, norm=0.49, scaled=26.40 | DIP: raw=-9.45, rel=-19.45, norm=0.79, scaled=85.00
RF | MCP: raw=-4.28, rel=-16.28, norm=0.25, scaled=9.56 | PIP: raw=13.83, rel=-6.17, norm=0.50, scaled=40.68 | DIP: raw=-27.11, rel=-37.11, norm=0.83, scaled=75.08
LF | MCP: raw=3.80, rel=1.80, norm=0.21, scaled=5.07 | PIP: raw=-3.50, rel=-3.50, norm=0.05, scaled=3.32 | DIP: raw=-3.59, rel=-48.59, norm=0.56, scaled=40.66
Frame 15827 => Movement: 4, Repetition: 1
  rh_FFJ1: 7.30
  rh_FFJ2: 18.63
  rh_FFJ3: 35.26
  rh_FFJ4: 0.00
  rh_LFJ1: 40.66
  rh_LFJ2: 3.32
  rh_LFJ3: 5.07
  rh_LFJ4: 0.00
  rh_LFJ5: -0.95
  rh_MFJ1: 85.00
  rh_MFJ2: 26.40
  rh_MFJ3: 7.00
  rh_MFJ4: 0.00
  rh_RFJ1: 75.08
  rh_RFJ2: 40.68
  rh_RFJ3: 9.56
  rh_RFJ4: 0.00
  rh_THJ1: 0.00
  rh_THJ2: 0.00
  rh_THJ3: 0.00
  rh_THJ4: 0.00
  rh_THJ5: 0.00
  rh_WRJ1: -7.27
  rh_WRJ2: 1.05
FF | MCP: raw=2.53, rel=-21.47, norm=0.48, scaled=35.55 | PIP: raw=-0.05, rel=-32.05, norm=0.21, scaled=18.68 | DIP: raw=-5.02, rel=-8.02, norm=0.07, scaled=7.25
MF | MCP: raw=-10.71, rel=-25.71, norm=0.20, scaled=7.31 | PIP: raw=2.86, rel=-17.14, norm=0.49, scaled=26.36 | DIP: raw=-9.30, rel=-19.30, norm=0.79, scaled=85.09
RF | MCP: raw=-4.10, rel=-16.10, norm=0.25, scaled=9.76 | PIP: raw=13.15, rel=-6.85, norm=0.50, scaled=40.44 | DIP: raw=-25.34, rel=-35.34, norm=0.84, scaled=75.53
LF | MCP: raw=4.47, rel=2.47, norm=0.22, scaled=5.33 | PIP: raw=-3.26, rel=-3.26, norm=0.05, scaled=3.43 | DIP: raw=-2.79, rel=-47.79, norm=0.57, scaled=41.26
Frame 15828 => Movement: 4, Repetition: 1
  rh_FFJ1: 7.25
  rh_FFJ2: 18.68
  rh_FFJ3: 35.55
  rh_FFJ4: 0.00
  rh_LFJ1: 41.26
  rh_LFJ2: 3.43
  rh_LFJ3: 5.33
  rh_LFJ4: 0.00
  rh_LFJ5: -0.92
  rh_MFJ1: 85.09
  rh_MFJ2: 26.36
  rh_MFJ3: 7.31
  rh_MFJ4: 0.00
  rh_RFJ1: 75.53
  rh_RFJ2: 40.44
  rh_RFJ3: 9.76
  rh_RFJ4: 0.00
  rh_THJ1: 0.00
  rh_THJ2: 0.00
  rh_THJ3: 0.00
  rh_THJ4: 0.00
  rh_THJ5: 0.00
  rh_WRJ1: -7.09
  rh_WRJ2: 1.09
FF | MCP: raw=2.77, rel=-21.23, norm=0.48, scaled=35.78 | PIP: raw=-0.05, rel=-32.05, norm=0.21, scaled=18.68 | DIP: raw=-5.02, rel=-8.02, norm=0.07, scaled=7.25
MF | MCP: raw=-10.21, rel=-25.21, norm=0.20, scaled=7.75 | PIP: raw=2.86, rel=-17.14, norm=0.49, scaled=26.36 | DIP: raw=-9.30, rel=-19.30, norm=0.79, scaled=85.09
RF | MCP: raw=-3.84, rel=-15.84, norm=0.26, scaled=10.04 | PIP: raw=12.72, rel=-7.28, norm=0.50, scaled=40.29 | DIP: raw=-23.74, rel=-33.74, norm=0.84, scaled=75.94
LF | MCP: raw=5.33, rel=3.33, norm=0.22, scaled=5.66 | PIP: raw=-3.02, rel=-3.02, norm=0.05, scaled=3.54 | DIP: raw=-1.79, rel=-46.79, norm=0.58, scaled=42.02
Frame 15829 => Movement: 4, Repetition: 1
  rh_FFJ1: 7.25
  rh_FFJ2: 18.68
  rh_FFJ3: 35.78
  rh_FFJ4: 0.00
  rh_LFJ1: 42.02
  rh_LFJ2: 3.54
  rh_LFJ3: 5.66
  rh_LFJ4: 0.00
  rh_LFJ5: -0.92
  rh_MFJ1: 85.09
  rh_MFJ2: 26.36
  rh_MFJ3: 7.75
  rh_MFJ4: 0.00
  rh_RFJ1: 75.94
  rh_RFJ2: 40.29
  rh_RFJ3: 10.04
  rh_RFJ4: 0.00
  rh_THJ1: 0.00
  rh_THJ2: 0.00
  rh_THJ3: 0.00
  rh_THJ4: 0.00
  rh_THJ5: 0.00
  rh_WRJ1: -7.09
  rh_WRJ2: 1.09
FF | MCP: raw=3.01, rel=-20.99, norm=0.49, scaled=36.00 | PIP: raw=-0.05, rel=-32.05, norm=0.21, scaled=18.68 | DIP: raw=-5.02, rel=-8.02, norm=0.07, scaled=7.25
MF | MCP: raw=-9.70, rel=-24.70, norm=0.21, scaled=8.19 | PIP: raw=2.86, rel=-17.14, norm=0.49, scaled=26.36 | DIP: raw=-9.30, rel=-19.30, norm=0.79, scaled=85.09
RF | MCP: raw=-3.58, rel=-15.58, norm=0.26, scaled=10.32 | PIP: raw=12.30, rel=-7.70, norm=0.50, scaled=40.14 | DIP: raw=-22.14, rel=-32.14, norm=0.85, scaled=76.35
LF | MCP: raw=6.18, rel=4.18, norm=0.22, scaled=5.99 | PIP: raw=-2.78, rel=-2.78, norm=0.05, scaled=3.65 | DIP: raw=-0.78, rel=-45.78, norm=0.59, scaled=42.77
Frame 15830 => Movement: 4, Repetition: 1
  rh_FFJ1: 7.25
  rh_FFJ2: 18.68
  rh_FFJ3: 36.00
  rh_FFJ4: 0.00
  rh_LFJ1: 42.77
  rh_LFJ2: 3.65
  rh_LFJ3: 5.99
  rh_LFJ4: 0.00
  rh_LFJ5: -0.92
  rh_MFJ1: 85.09
  rh_MFJ2: 26.36
  rh_MFJ3: 8.19
  rh_MFJ4: 0.00
  rh_RFJ1: 76.35
  rh_RFJ2: 40.14
  rh_RFJ3: 10.32
  rh_RFJ4: 0.00
  rh_THJ1: 0.00
  rh_THJ2: 0.00
  rh_THJ3: 0.00
  rh_THJ4: 0.00
  rh_THJ5: 0.00
  rh_WRJ1: -7.09
  rh_WRJ2: 1.09
FF | MCP: raw=3.26, rel=-20.74, norm=0.49, scaled=36.23 | PIP: raw=-0.05, rel=-32.05, norm=0.21, scaled=18.68 | DIP: raw=-5.02, rel=-8.02, norm=0.07, scaled=7.25
MF | MCP: raw=-9.19, rel=-24.19, norm=0.21, scaled=8.63 | PIP: raw=2.86, rel=-17.14, norm=0.49, scaled=26.36 | DIP: raw=-9.30, rel=-19.30, norm=0.79, scaled=85.09
RF | MCP: raw=-3.32, rel=-15.32, norm=0.26, scaled=10.61 | PIP: raw=11.88, rel=-8.12, norm=0.49, scaled=40.00 | DIP: raw=-20.54, rel=-30.54, norm=0.85, scaled=76.75
LF | MCP: raw=7.03, rel=5.03, norm=0.23, scaled=6.31 | PIP: raw=-2.55, rel=-2.55, norm=0.06, scaled=3.77 | DIP: raw=0.23, rel=-44.77, norm=0.60, scaled=43.53
Frame 15831 => Movement: 4, Repetition: 1
  rh_FFJ1: 7.25
  rh_FFJ2: 18.68
  rh_FFJ3: 36.23
  rh_FFJ4: 0.00
  rh_LFJ1: 43.53
  rh_LFJ2: 3.77
  rh_LFJ3: 6.31
  rh_LFJ4: 0.00
  rh_LFJ5: -0.92
  rh_MFJ1: 85.09
  rh_MFJ2: 26.36
  rh_MFJ3: 8.63
  rh_MFJ4: 0.00
  rh_RFJ1: 76.75
  rh_RFJ2: 40.00
  rh_RFJ3: 10.61
  rh_RFJ4: 0.00
  rh_THJ1: 0.00
  rh_THJ2: 0.00
  rh_THJ3: 0.00
  rh_THJ4: 0.00
  rh_THJ5: 0.00
  rh_WRJ1: -7.09
  rh_WRJ2: 1.09
FF | MCP: raw=3.96, rel=-20.04, norm=0.49, scaled=36.87 | PIP: raw=-0.05, rel=-32.05, norm=0.21, scaled=18.68 | DIP: raw=-5.02, rel=-8.02, norm=0.07, scaled=7.25
MF | MCP: raw=-8.92, rel=-23.92, norm=0.21, scaled=8.87 | PIP: raw=2.59, rel=-17.41, norm=0.49, scaled=26.26 | DIP: raw=-8.84, rel=-18.84, norm=0.79, scaled=85.36
RF | MCP: raw=-3.07, rel=-15.07, norm=0.26, scaled=10.89 | PIP: raw=11.07, rel=-8.93, norm=0.49, scaled=39.71 | DIP: raw=-18.94, rel=-28.94, norm=0.86, scaled=77.16
LF | MCP: raw=7.62, rel=5.62, norm=0.23, scaled=6.54 | PIP: raw=-2.31, rel=-2.31, norm=0.06, scaled=3.88 | DIP: raw=0.61, rel=-44.39, norm=0.61, scaled=43.82
Frame 15832 => Movement: 4, Repetition: 1
  rh_FFJ1: 7.25
  rh_FFJ2: 18.68
  rh_FFJ3: 36.87
  rh_FFJ4: 0.00
  rh_LFJ1: 43.82
  rh_LFJ2: 3.88
  rh_LFJ3: 6.54
  rh_LFJ4: 0.00
  rh_LFJ5: -0.85
  rh_MFJ1: 85.36
  rh_MFJ2: 26.26
  rh_MFJ3: 8.87
  rh_MFJ4: 0.00
  rh_RFJ1: 77.16
  rh_RFJ2: 39.71
  rh_RFJ3: 10.89
  rh_RFJ4: 0.00
  rh_THJ1: 0.00
  rh_THJ2: 0.00
  rh_THJ3: 0.00
  rh_THJ4: 0.00
  rh_THJ5: 0.00
  rh_WRJ1: -6.55
  rh_WRJ2: 1.20
