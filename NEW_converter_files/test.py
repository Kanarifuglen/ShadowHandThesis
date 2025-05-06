import h5py
import numpy as np
from tqdm import tqdm

def scan_for_nans(h5_path):
    with h5py.File(h5_path, 'r') as h5f:
        keys = h5f['movements'].keys()
        for key in tqdm(keys, desc="Scanning for NaNs"):
            data = h5f['movements'][key]['angles'][:]
            if np.isnan(data).any():
                print(f"🚨 Found NaNs in: {key}")

scan_for_nans("shadow_dataset.h5")
