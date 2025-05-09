"""
Discrete Cosine Transform (DCT) utility functions for motion prediction.

This module contains functions for DCT/IDCT calculations for motion prediction models.
"""

import math
import numpy as np
import torch

def build_dct_matrix(N):
    """Build a matrix to perform DCT via matrix multiplication.
    
    Based on implementation from Mao et al. 
    Source: https://github.com/wei-mao-2019/LearnTrajDep
    
    Args:
        N: Size of the DCT matrix
        
    Returns:
        Tensor containing the DCT transformation matrix
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
    """Build a matrix to perform inverse DCT via matrix multiplication.
    
    Args:
        N: Size of the IDCT matrix
        
    Returns:
        Tensor containing the IDCT transformation matrix
    """
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