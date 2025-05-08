# Improved SpatioTemporalTransformer.py - Hand Motion Prediction with DCT, Joint Embedding, and Memory Dictionary

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
#from scipy.fftpack import dct, idct
import json
import random 
import csv
import argparse
import pickle
from evaluation_utils import (  # Add this import
    plot_per_joint_error,
    plot_error_progression_by_frame,
    plot_action_comparison,
    generate_detailed_error_table,
    generate_joint_group_table,
    analyze_dct_coefficients,
    visualize_predictions
)

# === Config ===
T_OBS = 30  # Number of observed frames
T_PRED = 40  # Number of frames to predict
T_TOTAL = T_OBS + T_PRED
NUM_JOINTS = 24  # Total number of hand joints
MEMORY_CELLS = 16  # Increased memory cells for better pattern storage
# Groups joints based on hand structure for progressive decoding
PROGRESSIVE_STAGES = [
    [0], [1], [2, 3, 4, 5], [6, 7, 8, 9],
    [10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]
]
THUMB_JOINTS = [2, 3, 4, 5, 6]  # Joints that need special handling


#def apply_dct_2d(time_seq):
#    """"Apply DCT to a 2D temporal sequence along time dimension."""
#    time_seq = time_seq.astype(np.float32)  # Add this line
#    T, D = time_seq.shape
#    out = np.zeros_like(time_seq, dtype=np.float32)  
#    for d in range(D):
#        out[:, d] = dct(time_seq[:, d], type=2, norm='ortho')
#    return out

#def apply_idct_2d(freq_seq):
#    """Apply inverse DCT to a 2D frequency sequence along freq dimension."""
#    freq_seq = freq_seq.astype(np.float32)  
#    T, D = freq_seq.shape
#    out = np.zeros_like(freq_seq, dtype=np.float32)
#    for d in range(D):
#        out[:, d] = idct(freq_seq[:, d], type=2, norm='ortho')
#    return out


def build_dct_matrix(N):
    """Build a matrix to perform DCT via matrix multiplication.
    
    Based on implementation from Mao et al. 
    Source: https://github.com/wei-mao-2019/LearnTrajDep
    """
    dct_m = np.zeros((N, N))
    for k in range(N):
        for i in range(N):
            if k == 0:
                dct_m[k, i] = 1.0 / np.sqrt(N)
            else:
                dct_m[k, i] = np.sqrt(2.0/N) * np.cos(np.pi * k * (2 * i + 1) / (2 * N))
    return torch.from_numpy(dct_m.astype(np.float32))

def build_idct_matrix(N):
    """Build a matrix to perform inverse DCT via matrix multiplication."""
    mat = torch.zeros(N, N, dtype=torch.float32)
    for n in range(N):
        for k in range(N):
            alpha = torch.tensor(math.sqrt(1.0/N) if k == 0 else math.sqrt(2.0/N), 
                                dtype=torch.float32) 
            cos_val = torch.tensor(math.cos(math.pi*(2*n+1)*k/(2.0*N)), 
                                  dtype=torch.float32) 
            mat[n, k] = alpha * cos_val
    return mat

def batch_dct_transform(tensor, dct_m):
    """Apply DCT to a batch of sequences via matrix multiplication.
    
    Adapted from Mao et al. (2019) "Learning Trajectory Dependencies for Human Motion Prediction"
    Source: https://github.com/wei-mao-2019/LearnTrajDep
    
    Args:
        tensor: Input tensor of shape (B, N, D) where B is batch size,
               N is sequence length, and D is feature dimension
        dct_m: DCT transformation matrix of shape (N', N)
               
    Returns:
        DCT coefficients of shape (B, N', D)
    """
    B, N, D = tensor.shape
    
    # Reshape to apply transformation to each sequence in the batch
    x = tensor.reshape(-1, N).transpose(0, 1)  # (N, B*D)
    
    # Apply DCT matrix
    y = torch.matmul(dct_m, x)  # (N', B*D)
    
    # Reshape back to original batch format
    y = y.transpose(0, 1).reshape(B, -1, D)  # (B, N', D)
    
    return y

def batch_idct_transform(tensor, idct_m):
    """Apply IDCT to a batch of sequences via matrix multiplication.
    
    Adapted from Mao et al. (2019) "Learning Trajectory Dependencies for Human Motion Prediction"
    Source: https://github.com/wei-mao-2019/LearnTrajDep
    
    Args:
        tensor: Input DCT coefficients of shape (B, N, D)
        idct_m: IDCT transformation matrix of shape (N', N)
               
    Returns:
        Time domain signals of shape (B, N', D)
    """
    B, N, D = tensor.shape
    
    # Reshape to apply transformation
    x = tensor.reshape(-1, N).transpose(0, 1)  # (N, B*D)
    
    # Apply IDCT matrix
    y = torch.matmul(idct_m, x)  # (N', B*D)
    
    # Reshape back to original batch format
    y = y.transpose(0, 1).reshape(B, -1, D)  # (B, N', D)
    
    return y

def torch_idct_2d(freq, idct_mat):
    """Apply inverse DCT using matrix multiplication in PyTorch.
    
    Args:
        freq: Tensor of shape (B, T, D) containing DCT coefficients
        idct_mat: Matrix for IDCT transformation
        
    Returns:
        Tensor of shape (B, T, D) containing time-domain values
    """
    B, T, D = freq.shape
    
    # Create new output tensor (avoid in-place operations)
    result = torch.zeros(B, T, D, device=freq.device, dtype=freq.dtype)
    
    # Process each dimension separately
    for d in range(D):
        # Extract this dimension for all batches and timepoints: (B, T)
        freq_d = freq[:, :, d]
        
        # Apply IDCT matrix: (T, T) x (B, T, 1) -> (B, T)
        result[:, :, d] = torch.matmul(idct_mat, freq_d.unsqueeze(2)).squeeze(2)
    
    return result

# === Improved Dataset ===
class MovementDataset(Dataset):
    def __init__(self, h5_path, T_obs=T_OBS, T_pred=T_PRED, subset_size=None, 
                 prefiltered=True, filtered_keys=None):
        super().__init__()
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_total = T_obs + T_pred
        
        import h5py
        self.h5 = h5py.File(h5_path, 'r')
        
        # If using prefiltered keys, skip validation
        if prefiltered and filtered_keys is not None:
            self.valid_samples = filtered_keys
            print(f"🔎 Using pre-filtered dataset with {len(self.valid_samples)} segments")
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
        return len(self.valid_samples)
        
    def __getitem__(self, idx):
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

# === Improved Model Components ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """Improved positional encoding with more robust implementation."""
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return x

class MemoryDictionary(nn.Module):
    """Improved memory dictionary for better pattern storage and retrieval."""
    def __init__(self, num_joints, d_model, num_cells=MEMORY_CELLS):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.num_cells = num_cells
        
        # Learnable memory cells - one set per joint
        self.memory_keys = nn.Parameter(torch.randn(num_joints, num_cells, d_model) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(num_joints, num_cells, d_model) * 0.02)
        
        # Attention scaling factor
        self.scale = math.sqrt(d_model)
        
        # Per-joint query transformation
        self.query_transform = nn.Linear(d_model, d_model)
        
    def forward(self, joint_features):
        """
        Retrieve patterns from memory using attention.
        
        Args:
            joint_features: tensor of shape (B, J, D) containing features for each joint
            
        Returns:
            memory_patterns: tensor of shape (B, J, D) containing retrieved patterns
        """
        B, J, D = joint_features.shape
        
        # Transform input features to queries
        queries = self.query_transform(joint_features)  # (B, J, D)
        
        # Initialize output container
        memory_patterns = torch.zeros_like(joint_features)
        
        # Process each joint separately
        for j in range(J):
            # Get query for this joint
            query = queries[:, j, :]  # (B, D)
            
            # Get memory for this joint
            keys = self.memory_keys[j]    # (num_cells, D)
            values = self.memory_values[j]  # (num_cells, D)
            
            # Compute attention scores
            scores = torch.matmul(query, keys.transpose(0, 1)) / self.scale  # (B, num_cells)
            
            # Apply softmax to get attention weights
            attn_weights = torch.softmax(scores, dim=-1)  # (B, num_cells)
            
            # Retrieve from memory
            retrieved = torch.matmul(attn_weights, values)  # (B, D)
            
            # Store result
            memory_patterns[:, j, :] = retrieved
        
        return memory_patterns

class ProgressiveTransformerDecoder(nn.Module):
    """Improved progressive decoder following kinematic chain of the hand."""
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, progressive_stages):
        super().__init__()
        self.d_model = d_model
        self.progressive_stages = progressive_stages
        
        # Transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        # Stack multiple layers
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=num_layers
        )
        
        # Stage-specific projections to combine memory with decoder output
        self.stage_projections = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(len(progressive_stages))
        ])
        
    def forward(self, tgt, memory, memory_patterns, tgt_mask=None):
        """
        Forward pass for progressive decoding.
        
        Args:
            tgt: Decoder input tensor (B, T*J, D)
            memory: Encoder output tensor (B, T*J, D)
            memory_patterns: Patterns from memory dictionary (B, J, D)
            tgt_mask: Optional mask for decoder self-attention
            
        Returns:
            output: Decoder output tensor (B, T*J, D)
        """
        B, L, D = tgt.shape
        T = T_TOTAL
        J = NUM_JOINTS
        
        # Create a new target tensor at each stage (avoid in-place operations)
        current_tgt = tgt.clone()
        
        # Keep track of results for each stage
        all_stage_outputs = []
        
        # Collect all indices that will be updated
        all_updated_indices = []
        
        # Progressive decoding through stages
        for stage_idx, joint_indices in enumerate(self.progressive_stages):
            # Collect indices for current stage in the flattened tensor
            stage_indices = []
            for t in range(T):
                for j in joint_indices:
                    stage_indices.append(t * J + j)
            
            # Get target for current stage
            stage_indices_tensor = torch.tensor(stage_indices, device=tgt.device, dtype=torch.long)
            stage_tgt = current_tgt.index_select(1, stage_indices_tensor)
            
            # Decode using transformer
            stage_output = self.transformer_decoder(stage_tgt, memory, tgt_mask=tgt_mask)
            
            # Enhance with memory patterns for these joints
            memory_enhancement = []
            for j in joint_indices:
                # Extract memory patterns for this joint and expand to match time dimension
                joint_pattern = memory_patterns[:, j:j+1, :].expand(-1, T, -1)
                # Reshape to match stage output shape
                joint_pattern = joint_pattern.reshape(B, T, D)
                memory_enhancement.append(joint_pattern)
            
            # Concatenate memory enhancements for all joints in this stage
            memory_enhancement = torch.cat(memory_enhancement, dim=1)
            
            # Combined stage output with memory enhancement
            enhanced_output = torch.cat([stage_output, memory_enhancement], dim=-1)
            enhanced_output = self.stage_projections[stage_idx](enhanced_output)
            
            # Store stage output and indices for later composition
            all_stage_outputs.append(enhanced_output)
            all_updated_indices.append(stage_indices_tensor)
            
            # Update current_tgt for next stage (non-in-place)
            # Create mask for updated indices
            update_mask = torch.zeros(L, dtype=torch.bool, device=tgt.device)
            update_mask[stage_indices_tensor] = True
            
            # Create new target tensor combining updated and non-updated parts
            new_tgt = current_tgt.clone()
            for i, idx in enumerate(stage_indices):
                new_tgt[:, idx, :] = enhanced_output[:, i, :]
            
            current_tgt = new_tgt
        
        # Return final target tensor after all stages
        return current_tgt

class HandDCTMotionModel(nn.Module):
    """Improved transformer model for hand motion prediction with DCT."""
    def __init__(self, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, 
                 use_memory=True, progressive=True):
        super().__init__()
        self.use_memory = use_memory
        self.progressive = progressive
        self.d_model = d_model
        
        # Joint embeddings
        self.joint_embed = nn.Embedding(NUM_JOINTS, d_model)
        
        # DCT coefficient embeddings
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=T_TOTAL * NUM_JOINTS)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Memory dictionary
        if use_memory:
            self.memory_dict = MemoryDictionary(NUM_JOINTS, d_model, MEMORY_CELLS)
        
        # Progressive decoder
        if progressive:
            self.decoder = ProgressiveTransformerDecoder(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_layers=num_layers,
                progressive_stages=PROGRESSIVE_STAGES
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final projection to DCT coefficients
        self.decoder_proj = nn.Linear(d_model, 1)
        
        # Layer normalization before final projection
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass for hand motion prediction.
        
        Args:
            x: DCT input of shape (B, T, J)
            
        Returns:
            output: Predicted DCT residuals of shape (B, T, J)
        """
        B, T, J = x.shape  # (batch, time, joints)
        
        # Create joint embeddings: (1, 1, J) -> (B, T, J, d_model)
        joint_ids = torch.arange(J, device=x.device).unsqueeze(0).unsqueeze(0)
        joint_embs = self.joint_embed(joint_ids).expand(B, T, J, -1)
        
        # Project DCT input to embeddings: (B, T, J, 1) -> (B, T, J, d_model)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        
        # Combine input features with joint embeddings
        x = x + joint_embs  # (B, T, J, d_model)
        
        # Reshape to sequence: (B, T, J, d_model) -> (B, T*J, d_model)
        x = x.reshape(B, T * J, self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        memory = self.encoder(x)  # (B, T*J, d_model)
        
        # Prepare joint features for memory dictionary
        if self.use_memory:
            # Reshape memory to get per-joint features
            joint_features = memory.reshape(B, T, J, self.d_model).mean(dim=1)  # (B, J, d_model)
            
            # Query memory dictionary
            memory_patterns = self.memory_dict(joint_features)  # (B, J, d_model)
        else:
            memory_patterns = torch.zeros(B, J, self.d_model, device=x.device)
        
        # Decoding
        if self.progressive:
            output = self.decoder(x, memory, memory_patterns)
        else:
            # For non-progressive decoding, expand memory patterns to match sequence shape
            expanded_patterns = memory_patterns.unsqueeze(1).expand(-1, T, -1, -1).reshape(B, T*J, -1)
            output = self.decoder(x, memory) + expanded_patterns
            
        # Apply layer norm and final projection
        output = self.final_norm(output)
        output = self.decoder_proj(output).squeeze(-1).reshape(B, T, J)
        
        return output



# === Data Splitting Functions for Proper Evaluation ===
def create_subject_split(dataset, test_subjects=None, test_ratio=0.2):
    """Split dataset by subject_id"""
    # Get unique subjects
    unique_subjects = set()
    for key in dataset.valid_samples:
        subject = dataset.h5['movements'][key].attrs.get('subject_id', "Unknown")
        unique_subjects.add(subject)
    unique_subjects = list(unique_subjects)
    
    # If no test subjects provided, randomly select some
    if test_subjects is None:
        num_test_subjects = max(1, int(len(unique_subjects) * test_ratio))
        random.shuffle(unique_subjects)
        test_subjects = unique_subjects[:num_test_subjects]
    
    # Create the split
    train_keys = []
    test_keys = []
    
    for key in dataset.valid_samples:
        subject = dataset.h5['movements'][key].attrs.get('subject_id', "Unknown")
        if subject in test_subjects:
            test_keys.append(key)
        else:
            train_keys.append(key)
    
    print(f"Subject split: {len(train_keys)} training samples, {len(test_keys)} testing samples")
    print(f"Test subjects: {test_subjects}")
    return train_keys, test_keys

def create_movement_split(dataset, test_ratio=0.2):
    """Split dataset by movement_name/exercise_id"""
    # Get unique movement types
    movement_types = set()
    for key in dataset.valid_samples:
        movement = dataset.h5['movements'][key].attrs.get('exercise_id', "Unknown")
        movement_types.add(movement)
    
    # Randomly select test movements
    movement_types = list(movement_types)
    num_test_movements = max(1, int(len(movement_types) * test_ratio))
    random.shuffle(movement_types)
    test_movements = movement_types[:num_test_movements]
    
    # Create split
    train_keys = []
    test_keys = []
    for key in dataset.valid_samples:
        movement = dataset.h5['movements'][key].attrs.get('exercise_id', "Unknown")
        if movement in test_movements:
            test_keys.append(key)
        else:
            train_keys.append(key)
    
    print(f"Movement split: {len(train_keys)} training samples, {len(test_keys)} testing samples")
    print(f"Test movements: {test_movements}")
    return train_keys, test_keys

def create_cross_validation_splits(dataset, n_folds=5, by='subject_id'):
    """Create n-fold cross-validation splits by subject, movement, or session"""
    # Get unique values for the split criterion
    unique_values = set()
    for key in dataset.valid_samples:
        value = dataset.h5['movements'][key].attrs.get(by, "Unknown")
        unique_values.add(value)
    
    # Create folds
    unique_values = list(unique_values)
    random.shuffle(unique_values)
    fold_size = len(unique_values) // n_folds
    
    all_splits = []
    for i in range(n_folds):
        # Select test values for this fold
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else len(unique_values)
        test_values = unique_values[start_idx:end_idx]
        
        # Create train/test keys
        train_keys = []
        test_keys = []
        for key in dataset.valid_samples:
            value = dataset.h5['movements'][key].attrs.get(by, "Unknown")
            if value in test_values:
                test_keys.append(key)
            else:
                train_keys.append(key)
        
        all_splits.append((train_keys, test_keys))
    
    return all_splits


# === Training Function ===
def train_model(dataset, epochs=20, batch_size=8, learning_rate=1e-4, weight_decay=1e-5, 
                use_memory=True, progressive=True, save_path="../models/best_model.pth", device=None):
    """Improved training function with better hyperparameters and monitoring."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Disable anomaly detection in normal training to improve speed
    # torch.autograd.set_detect_anomaly(True)  # Commented out for speed
    
    # Create DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Reduce from 4 to 2 to reduce memory overhead
        pin_memory=True
    )
    
    # Create model
    model = HandDCTMotionModel(
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        use_memory=use_memory,
        progressive=progressive
    ).to(device)
    
    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5, 
        patience=5, 
        verbose=False
    )
    
    # Loss function
    l1_loss = nn.L1Loss() 
    
    # IDCT matrix for evaluation
    idct_mat = build_idct_matrix(T_PRED).to(device)
    
    # Training loop
    best_loss = float('inf')
    losses, maes = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_mae = 0, 0
        
        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as progress_bar:
            for obs_batch, fut_batch, _ in progress_bar:
                B = obs_batch.size(0)
                obs_batch = obs_batch.to(device)  # (B, T_OBS, J)
                fut_batch = fut_batch.to(device)  # (B, T_PRED, J)
                
                # Prepare input sequence with padding for future frames
                last_frame = obs_batch[:, -1:, :]  # (B, 1, J)
                padded_obs = torch.cat([
                    obs_batch, 
                    last_frame.repeat(1, T_PRED, 1)
                ], dim=1)  # (B, T_TOTAL, J)
                
                # Use batch DCT transform for more efficient processing
                dct_m = build_dct_matrix(T_TOTAL).to(device)
                input_dct = batch_dct_transform(padded_obs, dct_m)
                
                # Extract baseline DCT (last T_PRED coefficients)
                baseline_dct = input_dct[:, -T_PRED:, :].clone()  # (B, T_PRED, J)
                
                # Forward pass
                optimizer.zero_grad()
                predicted_residual = model(input_dct)  # (B, T_TOTAL, J)
                
                # Extract prediction residuals for future frames - use clone to avoid in-place issues
                pred_residual = predicted_residual[:, -T_PRED:, :].clone()  # (B, T_PRED, J)
                
                # Add residual to baseline - use addition instead of in-place add
                predicted_dct = pred_residual + baseline_dct  # (B, T_PRED, J)
                
                # Apply IDCT to get time domain prediction
                pred_fut = batch_idct_transform(predicted_dct, idct_mat)  # (B, T_PRED, J)
                
                # Create mask for thumb joints
                mask = torch.ones(pred_fut.shape[-1], dtype=torch.bool, device=device)
                mask[THUMB_JOINTS] = False
                
                # Apply mask
                masked_pred = pred_fut[:, :, mask]
                masked_target = fut_batch[:, :, mask]
                
                # Compute loss
                loss = l1_loss(masked_pred, masked_target)
                
                # Backpropagation
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Compute Mean Absolute Error
                with torch.no_grad():
                    mae = (masked_pred - masked_target).abs().mean().item()
                
                # Update statistics
                total_loss += loss.item()
                total_mae += mae
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'mae': f"{mae:.4f}"
                })
        
        # Calculate average metrics
        avg_loss = total_loss / len(loader)
        avg_mae = total_mae / len(loader)
        
        losses.append(avg_loss)
        maes.append(avg_mae)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, MAE={avg_mae:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model saved to {save_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss (L1)")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(maes)
    plt.title("Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (radians)")
    
    plt.tight_layout()
    plt.savefig("../plots/training_curves.png")
    print("📊 Saved training curves to '../plots/training_curves.png'")

    return model

def evaluate_model(model, dataset, device=None, batch_size=32):
    """Comprehensive evaluation protocol following Mao et al.
    
    Adapted from Mao et al. (2019) "Learning Trajectory Dependencies for Human Motion Prediction"
    Source: https://github.com/wei-mao-2019/LearnTrajDep/blob/master/evaluate.py
    
    Args:
        model: Trained motion prediction model
        dataset: Dataset containing test samples
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Precompute DCT and IDCT matrices
    dct_m = build_dct_matrix(T_TOTAL).to(device)
    idct_m = build_idct_matrix(T_PRED).to(device)
    
    # Define specific frames for evaluation (following standard papers)
    # 80ms, 160ms, 320ms, 400ms, 560ms, 1000ms at 25fps
    frames_per_second = 25
    ms_to_frames = lambda ms: int(ms * frames_per_second / 1000)
    eval_ms = [80, 160, 320, 400, 560, 1000]
    eval_frames = [ms_to_frames(ms) for ms in eval_ms]
    
    # Track metrics by movement/exercise type
    movement_results = {}
    
    # For additional analysis
    all_preds = []
    all_targets = []
    all_metadata = []
    sample_input_dct = None
    sample_baseline_dct = None
    sample_pred_residual = None
    
    # Define joint names (adjust according to your model)
    joint_names = [
        'Root_1', 'Root_2',
        'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Thumb_5',
        'Index_1', 'Index_2', 'Index_3', 'Index_4',
        'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4',
        'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Ring_5',
        'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'
    ]
    
    # Switch to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (obs_batch, fut_batch, metadata) in enumerate(tqdm(loader, desc="Evaluating")):
            B = obs_batch.size(0)
            obs_batch = obs_batch.to(device)
            fut_batch = fut_batch.to(device)
            
            # Process each sample in the batch
            for i in range(B):
                # Get movement type for categorization
                movement_type = str(metadata['exercise_id'][i])  # Convert to string for dictionary key
                if movement_type not in movement_results:
                    movement_results[movement_type] = {ms: [] for ms in eval_ms}
                
                # Standard prediction process
                obs = obs_batch[i].unsqueeze(0)
                fut = fut_batch[i].unsqueeze(0)
                
                # Prepare input with padding
                last_frame = obs[:, -1:, :]
                padded_obs = torch.cat([obs, last_frame.repeat(1, T_PRED, 1)], dim=1)
                
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
                
                # Apply mask for thumb joints if needed
                mask = torch.ones(pred_fut.shape[-1], dtype=torch.bool, device=device)
                mask[THUMB_JOINTS] = False
                
                # Save sample for additional analysis (first batch)
                if batch_idx == 0 and i == 0:
                    sample_input_dct = input_dct.clone().cpu()
                    sample_baseline_dct = baseline_dct.clone().cpu()
                    sample_pred_residual = pred_residual.clone().cpu()
                
                # Store predictions and targets for additional analysis
                if len(all_preds) < 10:  # Limit to 10 samples for memory efficiency
                    all_preds.append(pred_fut.clone().cpu())
                    all_targets.append(fut.clone().cpu())
                    all_metadata.append({k: v[i] for k, v in metadata.items()})
                
                # Calculate error at each evaluation frame
                for f_idx, frame in enumerate(eval_frames):
                    if frame < fut.shape[1]:  # Make sure frame exists
                        # MAE for angle representation
                        error = (pred_fut[0, frame, mask] - fut[0, frame, mask]).abs().mean().item()
                        movement_results[movement_type][eval_ms[f_idx]].append(error)
                
                # Visualization (limited to first few samples)
                if batch_idx < 3 and i == 0:
                    visualize_predictions(pred_fut[0], fut[0], batch_idx, metadata['movement_name'][i])
    
    # Calculate average metrics
    overall_results = {ms: [] for ms in eval_ms}
    for movement, metrics in movement_results.items():
        for ms, errors in metrics.items():
            if errors:
                avg_error = sum(errors) / len(errors)
                overall_results[ms].append(avg_error)
    
    # Compute overall averages
    overall_avg = {ms: sum(errors)/len(errors) if errors else float('nan') for ms, errors in overall_results.items()}
    
    # Print results in format similar to papers
    print("\n----- Evaluation Results (MAE) -----")
    header = ["Movement"] + [f"{ms}ms" for ms in eval_ms]
    print("  ".join(f"{h:<12}" for h in header))
    print("-" * 100)
    
    for movement, metrics in movement_results.items():
        row = [movement[:12]]
        for ms in eval_ms:
            if metrics[ms]:
                row.append(f"{sum(metrics[ms])/len(metrics[ms]):.4f}")
            else:
                row.append("N/A")
        print("  ".join(f"{r:<12}" for r in row))
    
    print("-" * 100)
    row = ["Average"]
    for ms in eval_ms:
        if overall_results[ms]:
            row.append(f"{overall_avg[ms]:.4f}")
        else:
            row.append("N/A")
    print("  ".join(f"{r:<12}" for r in row))
    
    # Export results to CSV for paper tables
    try:
        os.makedirs("../evaluations", exist_ok=True)
        with open('../evaluations/evaluation_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Movement'] + [f'{ms}ms' for ms in eval_ms])
            
            for movement, metrics in movement_results.items():
                row = [movement]
                for ms in eval_ms:
                    if metrics[ms]:
                        row.append(sum(metrics[ms]) / len(metrics[ms]))
                    else:
                        row.append('N/A')
                writer.writerow(row)
            
            # Add average row
            avg_row = ['Average']
            for ms in eval_ms:
                if overall_results[ms]:
                    avg_row.append(sum(overall_results[ms]) / len(overall_results[ms]))
                else:
                    avg_row.append('N/A')
            writer.writerow(avg_row)
            
        print(f"Results exported to ../evaluations/evaluation_results.csv")
    except Exception as e:
        print(f"Could not export results to CSV: {e}")
    
    # Additional analysis and visualizations
    print("\n----- Generating Additional Analysis -----")
    
    # Create directories for output
    os.makedirs("../plots", exist_ok=True)
    
    # Only proceed with additional analysis if we have data
    if all_preds and all_targets:
        try:
            # 1. Per-Joint Error Analysis
            try:
                from evaluation_utils import plot_per_joint_error
                # Stack first few predictions and targets
                stacked_preds = torch.cat(all_preds[:5], dim=0)
                stacked_targets = torch.cat(all_targets[:5], dim=0)
                # Create mask on CPU
                cpu_mask = torch.ones(stacked_preds.shape[-1], dtype=torch.bool)
                cpu_mask[THUMB_JOINTS] = False
                
                plot_per_joint_error(stacked_preds, stacked_targets, cpu_mask, joint_names)
                print("✅ Generated per-joint error analysis")
            except Exception as e:
                print(f"Could not generate per-joint error analysis: {e}")
                
            # 2. Error Progression Analysis
            try:
                from evaluation_utils import plot_error_progression_by_frame
                plot_error_progression_by_frame(stacked_preds, stacked_targets, cpu_mask)
                print("✅ Generated error progression analysis")
            except Exception as e:
                print(f"Could not generate error progression analysis: {e}")
            
            # 3. Action Comparison Visualization
            try:
                from evaluation_utils import plot_action_comparison
                plot_action_comparison(movement_results, eval_ms)
                print("✅ Generated action comparison visualization")
            except Exception as e:
                print(f"Could not generate action comparison: {e}")
            
            # 4. Detailed Error Table
            try:
                from evaluation_utils import generate_detailed_error_table
                generate_detailed_error_table(movement_results, eval_ms)
                print("✅ Generated detailed error table")
            except Exception as e:
                print(f"Could not generate detailed error table: {e}")
            
            # 5. Joint Group Analysis
            try:
                from evaluation_utils import generate_joint_group_table
                generate_joint_group_table(stacked_preds, stacked_targets, joint_names)
                print("✅ Generated joint group analysis")
            except Exception as e:
                print(f"Could not generate joint group analysis: {e}")
            
            # 6. DCT Coefficient Analysis
            if sample_input_dct is not None and sample_baseline_dct is not None and sample_pred_residual is not None:
                try:
                    from evaluation_utils import analyze_dct_coefficients
                    analyze_dct_coefficients(sample_input_dct, sample_baseline_dct, sample_pred_residual)
                    print("✅ Generated DCT coefficient analysis")
                except Exception as e:
                    print(f"Could not generate DCT coefficient analysis: {e}")
        
        except Exception as e:
            print(f"Error during additional analysis: {e}")
    else:
        print("⚠️ No samples available for additional analysis")
    
    # Return the computed metrics
    return {
        "by_movement": movement_results,
        "overall": overall_avg
    }



def main():
    """Main function to train and test the model."""
    parser = argparse.ArgumentParser(description="Train and test the hand motion prediction model")
    parser.add_argument('--data', type=str, default="../datasets/shadow_dataset.h5", help="Path to the dataset")
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both', help="Operation mode")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--subset-size', type=int, default=None, help="Use a subset of data for debugging")
    parser.add_argument('--no-memory', action='store_true', help="Disable memory dictionary")
    parser.add_argument('--no-progressive', action='store_true', help="Disable progressive decoding")
    parser.add_argument('--model-path', type=str, default="../models/best_model.pth", help="Path for saving/loading model")
    
    # Dataset filtering options
    parser.add_argument('--save-dataset', action='store_true', help="Save filtered HDF5 movement keys")
    parser.add_argument('--load-dataset', action='store_true', help="Load filtered HDF5 movement keys")
    parser.add_argument('--filtered-keys', type=str, default="filtered_keys.pkl", help="Path for saving/loading filtered keys")
    parser.add_argument('--filter-save', action='store_true', help="Filter and save a clean dataset")
    parser.add_argument('--filtered-output', type=str, default="../datasets/shadow_dataset_filtered.h5", 
                   help="Output path for filtered dataset")
    # New data splitting parameters
    parser.add_argument('--split-type', choices=['subject', 'movement', 'random'], default='subject',
                      help="How to split the dataset for train/test (by subject, movement, or random)")
    parser.add_argument('--test-ratio', type=float, default=0.2, 
                      help="Percentage of data to use for testing (for random split)")
    parser.add_argument('--test-subjects', type=str, nargs='+', 
                      help="Specific subjects to use for testing (for subject split)")
    parser.add_argument('--cross-val', action='store_true', help="Use cross-validation")
    parser.add_argument('--n-folds', type=int, default=5, help="Number of folds for cross-validation")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if we should load prefiltered keys
    filtered_keys = None
    if args.load_dataset and os.path.exists(args.filtered_keys):
        print(f"📂 Loading prefiltered keys from {args.filtered_keys}")
        with open(args.filtered_keys, 'rb') as f:
            filtered_keys = pickle.load(f)
        print(f"Found {len(filtered_keys)} prefiltered keys")
    
    # Create dataset
    dataset = MovementDataset(
        h5_path=args.data,
        T_obs=T_OBS,
        T_pred=T_PRED,
        subset_size=args.subset_size,
        prefiltered=(args.load_dataset and filtered_keys is not None),
        filtered_keys=filtered_keys
    )
    
    # Save filtered keys if requested
    if args.save_dataset:
        print(f"💾 Saving filtered keys to {args.filtered_keys}")
        with open(args.filtered_keys, 'wb') as f:
            pickle.dump(dataset.valid_samples, f)
        print(f"✅ Saved {len(dataset.valid_samples)} filtered keys")
    
    # Filter and save dataset if requested  
    if args.filter_save:
        print(f"💾 Saving filtered dataset to {args.filtered_output} ...")
        import h5py
        with h5py.File(args.data, 'r') as h5_in, h5py.File(args.filtered_output, 'w') as h5_out:
            in_grp = h5_in['movements']
            out_grp = h5_out.create_group('movements')
            for key in dataset.valid_samples:
                in_grp.copy(key, out_grp)
        print(f"✅ Filtered dataset saved to {args.filtered_output}")
    
    # Create data splits for train/test
    train_keys, test_keys = None, None
    
    if args.cross_val:
        # Create cross-validation splits
        print(f"Creating {args.n_folds}-fold cross-validation splits by {args.split_type}...")
        cv_splits = create_cross_validation_splits(dataset, n_folds=args.n_folds, by=args.split_type + '_id')
        # For simplicity, use first fold for now (can be extended to use all folds)
        train_keys, test_keys = cv_splits[0]
        print(f"Using fold 1/{args.n_folds} with {len(train_keys)} train samples and {len(test_keys)} test samples")
    else:
        # Create a single train/test split
        if args.split_type == 'subject':
            train_keys, test_keys = create_subject_split(dataset, test_subjects=args.test_subjects, test_ratio=args.test_ratio)
        elif args.split_type == 'movement':
            train_keys, test_keys = create_movement_split(dataset, test_ratio=args.test_ratio)
        else:  # random split
            # Implement a simple random split
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            split = int(len(indices) * (1 - args.test_ratio))
            train_indices = indices[:split]
            test_indices = indices[split:]
            train_keys = [dataset.valid_samples[i] for i in train_indices]
            test_keys = [dataset.valid_samples[i] for i in test_indices]
            print(f"Random split: {len(train_keys)} training samples, {len(test_keys)} testing samples")
    
    # Create train and test datasets
    train_dataset = MovementDataset(
        h5_path=args.data,
        T_obs=T_OBS,
        T_pred=T_PRED,
        prefiltered=True,
        filtered_keys=train_keys
    )
    
    test_dataset = MovementDataset(
        h5_path=args.data,
        T_obs=T_OBS,
        T_pred=T_PRED,
        prefiltered=True,
        filtered_keys=test_keys
    )
    
    # Train model
    if args.mode in ['train', 'both']:
        model = train_model(
            dataset=train_dataset,  # Use training dataset for training
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_memory=not args.no_memory,
            progressive=not args.no_progressive,
            save_path=args.model_path,
            device=device
        )
    
    # Test model using the improved evaluation function
    if args.mode in ['test', 'both']:
        if args.mode == 'test':
            # Create model and load weights
            model = HandDCTMotionModel(
                use_memory=not args.no_memory,
                progressive=not args.no_progressive
            ).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Use the improved evaluation function with test dataset
        results = evaluate_model(
            model=model, 
            dataset=test_dataset,  # Use test dataset for evaluation 
            device=device, 
            batch_size=args.batch_size
        )
        
        with open("../evaluations/evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Saved ../evaluations/evaluation_results.json")
        

if __name__ == "__main__":
    main()