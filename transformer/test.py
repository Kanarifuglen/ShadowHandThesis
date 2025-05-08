import sys
import numpy as np
import h5py
import os

def test_h5(file_path):
    print(f"🔍 Testing and cleaning .h5 dataset: {file_path}")
    
    # We'll create a temporary file first
    tmp_file = file_path + ".tmp"
    
    with h5py.File(file_path, "r") as fin, h5py.File(tmp_file, "w") as fout:
        movements_in = fin["movements"]
        movements_out = fout.create_group("movements")
        
        num_groups = len(movements_in)
        print(f"➡️ Found {num_groups} movement groups.")
        
        invalid_lengths = []
        nan_count = 0
        zero_count = 0
        unique_movements = set()
        kept_count = 0
        
        for key in movements_in:
            grp = movements_in[key]
            angles = grp["angles"][:]
            valid_length = grp.attrs.get("valid_length", angles.shape[0])
            movement_id = grp.attrs.get("movement_id", -1)
            unique_movements.add(movement_id)
            
            if angles.shape[0] != valid_length:
                invalid_lengths.append(key)
            
            has_nan = np.isnan(angles).any()
            is_zero = np.all(angles == 0)
            
            if has_nan:
                nan_count += 1
                print(f"❌ NaNs found in group: {key}")
            elif is_zero:
                zero_count += 1
                print(f"⚠️ All-zero angles found in group: {key}")
            else:
                # copy valid group
                new_grp = movements_out.create_group(key)
                new_grp.create_dataset("angles", data=angles)
                for attr_key in grp.attrs:
                    new_grp.attrs[attr_key] = grp.attrs[attr_key]
                kept_count += 1
        
        print(f"✅ Kept {kept_count} valid groups.")
        
        if invalid_lengths:
            print(f"⚠️ Groups with mismatched valid_length: {invalid_lengths}")
        else:
            print(f"✅ All groups match valid_length.")
        
        if nan_count > 0:
            print(f"❌ Removed {nan_count} groups containing NaNs.")
        else:
            print(f"✅ No NaNs found in dataset.")
        
        if zero_count > 0:
            print(f"⚠️ Removed {zero_count} all-zero movement groups.")
        else:
            print(f"✅ No all-zero movement groups found.")
        
        print(f"📝 Contains {len(unique_movements)} unique movement IDs: {sorted(unique_movements)}")
    
    # Replace original file
    os.replace(tmp_file, file_path)
    print(f"🎉 Cleaned dataset overwritten in: {file_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <file.h5>")
        return
    
    file_path = sys.argv[1]
    
    if file_path.endswith(".h5"):
        test_h5(file_path)
    else:
        print("❌ Unsupported file type. Must be .h5")

if __name__ == "__main__":
    main()
