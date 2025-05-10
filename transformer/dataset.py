"""
Dataset classes for hand motion prediction.

This module contains the MovementDataset class for loading and processing
hand movement data from HDF5 files, plus functions for simulator visualization.
"""
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
from tqdm import tqdm

# Import configuration
from config import T_OBS, T_PRED, T_TOTAL, NUM_JOINTS, JOINT_NAMES

# Import DCT utilities
from dct_utils import (
    build_dct_matrix, 
    build_idct_matrix,
    batch_dct_transform, 
    batch_idct_transform
)

class MovementDataset(Dataset):
    """Dataset for loading hand movement data from HDF5 files."""
    
    def __init__(self, h5_path, T_obs=T_OBS, T_pred=T_PRED, subset_size=None, 
                 prefiltered=True, filtered_keys=None, dataset_type=None):
        """Initialize the dataset.
        
        Args:
            h5_path: Path to HDF5 file containing movement data
            T_obs: Number of observed frames
            T_pred: Number of frames to predict
            subset_size: Optional size limit for the dataset
            prefiltered: Whether to use prefiltered movement keys
            filtered_keys: List of prefiltered movement keys
            dataset_type: Optional type descriptor (e.g., "training", "test")
        """
        super().__init__()
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_total = T_obs + T_pred
        
        import h5py
        self.h5 = h5py.File(h5_path, 'r')
        
        # If using prefiltered keys, skip validation
        if prefiltered and filtered_keys is not None:
            self.valid_samples = filtered_keys
            type_label = f"{dataset_type} " if dataset_type else ""
            print(f"🔎 Using pre-filtered {type_label}dataset with {len(self.valid_samples)} segments")
        else:
            all_keys = list(self.h5['movements'].keys())
            print(f"🔎 Total segments in HDF5: {len(all_keys)}")
            
            # Improved validation: Check for NaNs and sufficient length
            valid_keys = []
            for k in all_keys:
                seq = self.h5['movements'][k]['angles'][:]
                valid_length = self.h5['movements'][k].attrs['valid_length']
                
                if not np.isnan(seq).any() and valid_length >= self.T_total:
                    valid_keys.append(k)
                else:
                    reason = "NaNs present" if np.isnan(seq).any() else "insufficient length"
                    print(f"🚫 Skipping {k} due to {reason}")
                    
            self.valid_samples = valid_keys
            print(f"✅ Validated {len(self.valid_samples)} clean segments")
            
        if subset_size:
            self.valid_samples = self.valid_samples[:subset_size]
            print(f"Using subset of {len(self.valid_samples)} segments")
        
        # Pre-compute statistics for normalization
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute mean and std for normalization."""
        all_angles = []
        for k in self.valid_samples[:min(100, len(self.valid_samples))]:  # Use subset for efficiency
            angles = self.h5['movements'][k]['angles'][:]
            all_angles.append(angles)
        
        all_angles = np.concatenate(all_angles, axis=0)
        self.mean = np.mean(all_angles, axis=0)
        self.std = np.std(all_angles, axis=0)
        self.std[self.std < 1e-5] = 1.0  # Avoid division by zero
            
    def __len__(self):
        """Return the number of valid samples in the dataset."""
        return len(self.valid_samples)
        
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (observed sequence, future sequence, metadata)
        """
        key = self.valid_samples[idx]
        grp = self.h5['movements'][key]
        angles = grp['angles'][:].astype(np.float32)
        valid_length = grp.attrs['valid_length']
        
        angles = angles[:valid_length]
        
        # Normalize
        angles = (angles - self.mean) / self.std
        
        # Get observation and future sequences
        obs = angles[:self.T_obs]
        fut = angles[self.T_obs:self.T_total]

        obs = torch.tensor(obs, dtype=torch.float32)
        fut = torch.tensor(fut, dtype=torch.float32)
        
        # Safety check: pad if needed
        if obs.shape[0] < self.T_obs:
            pad_len = self.T_obs - obs.shape[0]
            obs = np.pad(obs, ((0, pad_len), (0, 0)), mode='edge')  # Use edge padding instead of zeros
            
        if fut.shape[0] < self.T_pred:
            pad_len = self.T_pred - fut.shape[0]
            fut = np.pad(fut, ((0, pad_len), (0, 0)), mode='edge')  # Use edge padding
            
        assert obs.shape == (self.T_obs, NUM_JOINTS), f"{key}: Expected obs shape {(self.T_obs, NUM_JOINTS)}, got {obs.shape}"
        assert fut.shape == (self.T_pred, NUM_JOINTS), f"{key}: Expected fut shape {(self.T_pred, NUM_JOINTS)}, got {fut.shape}"
        
        metadata = {
            'movement_id': grp.attrs.get('movement_id', 0),
            'movement_name': grp.attrs.get('movement_name', "Unknown"),
            'session_id': grp.attrs.get('session_id', "Unknown"),
            'subject_id': grp.attrs.get('subject_id', "Unknown"),
            'exercise_id': grp.attrs.get('exercise_id', "Unknown"),
            'exercise_table': grp.attrs.get('exercise_table', "Unknown"),
            'repetition_id': grp.attrs.get('repetition_id', 0),
        }
        
        return obs, fut, metadata
    

class CombinedDataset(Dataset):
    """Dataset for combining multiple movement datasets with balanced sampling."""
    
    def __init__(self, datasets, dataset_names=None):
        """Initialize the combined dataset.
        
        Args:
            datasets: List of MovementDataset objects to combine
            dataset_names: Optional list of names for each dataset
        """
        super().__init__()
        self.datasets = datasets
        self.dataset_names = dataset_names or [f"dataset_{i}" for i in range(len(datasets))]
        
        # Calculate dataset sizes and weights
        self.sizes = [len(ds) for ds in datasets]
        self.total_size = sum(self.sizes)
        
        # Calculate weights to balance datasets
        max_size = max(self.sizes)
        self.weights = [max_size / size for size in self.sizes]
        
        # Create dataset indices for tracking source
        self.dataset_indices = []
        for i, size in enumerate(self.sizes):
            self.dataset_indices.extend([i] * size)
        
        print(f"Created combined dataset with {self.total_size} samples:")
        for i, (name, size, weight) in enumerate(zip(self.dataset_names, self.sizes, self.weights)):
            print(f"  {name}: {size} samples, weight: {weight:.2f}")
    
    def __len__(self):
        """Return the total number of samples."""
        return self.total_size
    
    def __getitem__(self, idx):
        """Get a sample based on the combined index.
        
        Args:
            idx: Combined dataset index
            
        Returns:
            tuple: (observed sequence, future sequence, metadata)
        """
        # Get source dataset index
        dataset_idx = self.dataset_indices[idx]
        
        # Calculate index within the source dataset
        local_idx = idx
        for i in range(dataset_idx):
            local_idx -= self.sizes[i]
        
        # Get item from source dataset
        obs, fut, metadata = self.datasets[dataset_idx][local_idx]
        
        # Add source dataset to metadata
        metadata['source_dataset'] = self.dataset_names[dataset_idx]
        
        return obs, fut, metadata
    
    def get_sample_weights(self):
        """Get sample weights for balanced sampling.
        
        Returns:
            torch.Tensor: Weights for each sample
        """
        sample_weights = torch.ones(self.total_size)
        start_idx = 0
        
        for i, size in enumerate(self.sizes):
            sample_weights[start_idx:start_idx + size] = self.weights[i]
            start_idx += size
            
        return sample_weights

def create_simulator_dataset(model, dataset, output_path, num_samples=100, device=None):
    """
    Generate and save predicted movements for simulator visualization.
    
    Args:
        model: Trained motion prediction model
        dataset: Dataset containing test samples
        output_path: Path to save the simulator-compatible dataset
        num_samples: Number of sequences to generate (default: 100)
        device: Device to run inference on
        
    Returns:
        Path to the created dataset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create DataLoader with fixed seed for reproducibility
    torch.manual_seed(42)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Precompute DCT and IDCT matrices
    dct_m = build_dct_matrix(T_TOTAL).to(device)
    idct_m = build_idct_matrix(T_PRED).to(device)
    
    # Create output directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create HDF5 file for storing predictions
    with h5py.File(output_path, 'w') as h5out:
        # Create main groups
        pred_group = h5out.create_group('predictions')
        gt_group = h5out.create_group('ground_truth')
        metadata_group = h5out.create_group('metadata')
        
        # Track processed samples
        count = 0
        
        # Switch to evaluation mode
        model.eval()
        
        with torch.no_grad():
            for obs_batch, fut_batch, metadata in tqdm(loader, desc="Generating predictions"):
                if count >= num_samples:
                    break
                
                # Process batch
                obs_batch = obs_batch.to(device)
                fut_batch = fut_batch.to(device)
                
                # Standard prediction process
                last_frame = obs_batch[:, -1:, :]
                padded_obs = torch.cat([obs_batch, last_frame.repeat(1, T_PRED, 1)], dim=1)
                
                # Apply batch DCT
                input_dct = batch_dct_transform(padded_obs, dct_m)
                
                # Forward pass
                predicted_residual = model(input_dct)
                
                # Extract future part
                pred_residual = predicted_residual[:, -T_PRED:, :]
                baseline_dct = input_dct[:, -T_PRED:, :]
                predicted_dct = pred_residual + baseline_dct
                
                # Convert back to time domain
                pred_fut = batch_idct_transform(predicted_dct, idct_m)
                
                # Combine observation and prediction for full sequence
                full_sequence = torch.cat([obs_batch, pred_fut], dim=1)
                
                # Apply denormalization if needed (use dataset statistics)
                full_sequence_denorm = full_sequence * dataset.std + dataset.mean
                fut_denorm = fut_batch * dataset.std + dataset.mean
                
                # Convert to numpy for storage
                pred_np = full_sequence_denorm.cpu().numpy()[0]  # [0] to remove batch dimension
                gt_np = torch.cat([obs_batch, fut_batch], dim=1).cpu().numpy()[0] * dataset.std + dataset.mean
                
                # Store in HDF5 file
                pred_group.create_dataset(f'seq_{count}', data=pred_np)
                gt_group.create_dataset(f'seq_{count}', data=gt_np)
                
                # Store metadata
                meta_grp = metadata_group.create_group(f'seq_{count}')
                for key, value in metadata.items():
                    if isinstance(value, torch.Tensor):
                        meta_grp.attrs[key] = value.item() if value.numel() == 1 else value[0].item()
                    else:
                        meta_grp.attrs[key] = value
                
                count += 1
        
        # Add global metadata
        h5out.attrs['model_name'] = type(model).__name__
        h5out.attrs['date_created'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h5out.attrs['num_sequences'] = count
        h5out.attrs['joint_names'] = np.array(JOINT_NAMES, dtype='S')
    
    print(f"✅ Created simulator dataset with {count} sequences at: {output_path}")
    return output_path



def load_simulator_data(h5_path, sequence_id=0, use_predicted=True):
    """
    Load a specific sequence from the simulator dataset.
    
    Args:
        h5_path: Path to the simulator dataset
        sequence_id: ID of the sequence to load
        use_predicted: Whether to load predicted (True) or ground truth (False) sequence
        
    Returns:
        motion_data: Motion data formatted for the simulator
        metadata: Sequence metadata
    """
    with h5py.File(h5_path, 'r') as f:
        # Choose prediction or ground truth
        group_name = 'predictions' if use_predicted else 'ground_truth'
        
        # Load sequence data
        sequence_key = f'seq_{sequence_id}'
        if sequence_key not in f[group_name]:
            raise ValueError(f"Sequence {sequence_id} not found in {group_name}")
        
        motion_data = f[group_name][sequence_key][:]
        
        # Load metadata
        metadata = {}
        if sequence_key in f['metadata']:
            meta_group = f['metadata'][sequence_key]
            for key in meta_group.attrs:
                metadata[key] = meta_group.attrs[key]
    
    return motion_data, metadata