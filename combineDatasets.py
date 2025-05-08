class CombinedMovementDataset(Dataset):
    """Dataset class that combines multiple HDF5 datasets."""
    
    def __init__(self, dataset_paths, T_obs=T_OBS, T_pred=T_PRED, subset_size=None):
        """
        Initialize a combined dataset from multiple HDF5 files.
        
        Args:
            dataset_paths: List of paths to HDF5 datasets
            T_obs: Number of frames to observe
            T_pred: Number of frames to predict
            subset_size: Optional limit on total dataset size
        """
        super().__init__()
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_total = T_obs + T_pred
        
        # Store dataset sources for each sample
        self.dataset_sources = []
        
        # Store h5 handles
        self.h5_files = []
        
        # Collect valid samples from all datasets
        self.valid_samples = []
        total_segments = 0
        
        for dataset_path in dataset_paths:
            print(f"Loading dataset: {dataset_path}")
            h5 = h5py.File(dataset_path, 'r')
            self.h5_files.append(h5)
            
            # Get all keys
            all_keys = list(h5['movements'].keys())
            total_segments += len(all_keys)
            
            # Validate samples
            valid_keys = []
            for k in all_keys:
                seq = h5['movements'][k]['angles'][:]
                valid_length = h5['movements'][k].attrs['valid_length']
                
                if not np.isnan(seq).any() and valid_length >= self.T_total:
                    valid_keys.append(k)
                    # Store which dataset this sample came from
                    self.dataset_sources.append(len(self.h5_files) - 1)
            
            # Add valid keys with their source dataset index
            self.valid_samples.extend([(len(self.h5_files) - 1, k) for k in valid_keys])
            
            print(f"  Found {len(valid_keys)} valid segments in {dataset_path}")
        
        print(f"🔎 Total segments across all datasets: {total_segments}")
        print(f"✅ Using {len(self.valid_samples)} valid segments")
        
        # Limit dataset size if requested
        if subset_size and subset_size < len(self.valid_samples):
            # Randomly sample to maintain distribution
            indices = np.random.choice(
                len(self.valid_samples), 
                size=subset_size, 
                replace=False
            )
            self.valid_samples = [self.valid_samples[i] for i in indices]
            self.dataset_sources = [self.dataset_sources[i] for i in indices]
            print(f"🔍 Using subset of {len(self.valid_samples)} segments")
        
        # Compute dataset statistics for normalization
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute mean and std for normalization."""
        all_angles = []
        sample_limit = min(1000, len(self.valid_samples))  # Use max 1000 samples for stats
        
        for i in range(sample_limit):
            dataset_idx, key = self.valid_samples[i]
            h5 = self.h5_files[dataset_idx]
            angles = h5['movements'][key]['angles'][:]
            valid_length = h5['movements'][key].attrs['valid_length']
            all_angles.append(angles[:valid_length])
        
        all_angles = np.concatenate(all_angles, axis=0)
        self.mean = np.mean(all_angles, axis=0)
        self.std = np.std(all_angles, axis=0)
        self.std[self.std < 1e-5] = 1.0  # Avoid division by zero
        
        print(f"📊 Computed statistics from {len(all_angles)} frames")
        
    def __len__(self):
        return len(self.valid_samples)
        
    def __getitem__(self, idx):
        dataset_idx, key = self.valid_samples[idx]
        h5 = self.h5_files[dataset_idx]
        grp = h5['movements'][key]
        
        angles = grp['angles'][:].astype(np.float32)
        valid_length = grp.attrs['valid_length']
        
        angles = angles[:valid_length]
        
        # Normalize
        angles = (angles - self.mean) / self.std
        
        # Get observation and future sequences
        obs = angles[:self.T_obs]
        fut = angles[self.T_obs:self.T_total]
        
        # Safety check: pad if needed
        if obs.shape[0] < self.T_obs:
            pad_len = self.T_obs - obs.shape[0]
            obs = np.pad(obs, ((0, pad_len), (0, 0)), mode='edge')
            
        if fut.shape[0] < self.T_pred:
            pad_len = self.T_pred - fut.shape[0]
            fut = np.pad(fut, ((0, pad_len), (0, 0)), mode='edge')
            
        assert obs.shape == (self.T_obs, NUM_JOINTS), f"Expected obs shape {(self.T_obs, NUM_JOINTS)}, got {obs.shape}"
        assert fut.shape == (self.T_pred, NUM_JOINTS), f"Expected fut shape {(self.T_pred, NUM_JOINTS)}, got {fut.shape}"
        
        metadata = {
            'movement_id': grp.attrs.get('movement_id', 0),
            'movement_name': grp.attrs.get('movement_name', "Unknown"),
            'session_id': grp.attrs.get('session_id', "Unknown"),
            'subject_id': grp.attrs.get('subject_id', "Unknown"),
            'exercise_id': grp.attrs.get('exercise_id', "Unknown"),
            'exercise_table': grp.attrs.get('exercise_table', "Unknown"),
            'repetition_id': grp.attrs.get('repetition_id', 0),
            'dataset_source': dataset_idx  # Track which dataset this came from
        }
        
        return obs, fut, metadata
    
    def get_train_val_split(self, val_ratio=0.1, by_source=False, val_source_idx=None):
        """
        Split the dataset into training and validation sets.
        
        Args:
            val_ratio: Fraction of data to use for validation (ignored if by_source=True)
            by_source: If True, split based on dataset source rather than random
            val_source_idx: If provided, use this dataset index for validation
            
        Returns:
            train_indices, val_indices
        """
        if by_source:
            if val_source_idx is None:
                # Use the last dataset as validation by default
                val_source_idx = len(self.h5_files) - 1
                
            train_indices = [i for i, (src, _) in enumerate(self.valid_samples) 
                              if src != val_source_idx]
            val_indices = [i for i, (src, _) in enumerate(self.valid_samples) 
                            if src == val_source_idx]
        else:
            # Random split
            indices = np.arange(len(self.valid_samples))
            np.random.shuffle(indices)
            split = int(val_ratio * len(indices))
            train_indices = indices[split:].tolist()
            val_indices = indices[:split].tolist()
            
        return train_indices, val_indices
    
    def get_split_by_movement(self, test_movements):
        """
        Split the dataset based on movement types.
        
        Args:
            test_movements: List of movement IDs to use for testing
            
        Returns:
            train_indices, test_indices
        """
        train_indices = []
        test_indices = []
        
        for i in range(len(self.valid_samples)):
            _, _, metadata = self[i]
            if metadata['movement_id'] in test_movements:
                test_indices.append(i)
            else:
                train_indices.append(i)
                
        return train_indices, test_indices
    
    def get_split_by_subject(self, test_subjects):
        """
        Split the dataset based on subjects.
        
        Args:
            test_subjects: List of subject IDs to use for testing
            
        Returns:
            train_indices, test_indices
        """
        train_indices = []
        test_indices = []
        
        for i in range(len(self.valid_samples)):
            _, _, metadata = self[i]
            if metadata['subject_id'] in test_subjects:
                test_indices.append(i)
            else:
                train_indices.append(i)
                
        return train_indices, test_indices
        
    def close(self):
        """Close all open H5 files."""
        for h5 in self.h5_files:
            h5.close()