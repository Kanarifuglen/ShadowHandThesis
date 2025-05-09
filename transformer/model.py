"""
Hand motion prediction model with DCT transformation and progressive decoding.

This module contains the model architecture for hand motion prediction.
"""

import torch
import torch.nn as nn
import math

from config import (
    T_TOTAL, NUM_JOINTS, MEMORY_CELLS, PROGRESSIVE_STAGES
)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Input tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class MemoryDictionary(nn.Module):
    """Memory dictionary for pattern storage and retrieval."""
    
    def __init__(self, num_joints, d_model, num_cells=MEMORY_CELLS):
        """Initialize the memory dictionary.
        
        Args:
            num_joints: Number of joints
            d_model: Model dimension
            num_cells: Number of memory cells per joint
        """
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
        """Retrieve patterns from memory using attention.
        
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
    """Progressive decoder following kinematic chain of the hand."""
    
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, progressive_stages):
        """Initialize the progressive decoder.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            num_layers: Number of transformer layers
            progressive_stages: List of joint groups for progressive decoding
        """
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
        """Forward pass for progressive decoding.
        
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
    """Transformer model for hand motion prediction with DCT."""
    
    def __init__(self, d_model=128, nhead=8, num_layers=3, dim_feedforward=512):
        """Initialize the hand motion model.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            use_memory: Whether to use memory dictionary
            progressive: Whether to use progressive decoding
        """
        super().__init__()
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
       
        self.memory_dict = MemoryDictionary(NUM_JOINTS, d_model, MEMORY_CELLS)
        
        # Progressive decoder
        
        self.decoder = ProgressiveTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            progressive_stages=PROGRESSIVE_STAGES
            )
        # Final projection to DCT coefficients
        self.decoder_proj = nn.Linear(d_model, 1)
        
        # Layer normalization before final projection
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """Forward pass for hand motion prediction.
        
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
        
        # Reshape memory to get per-joint features
        joint_features = memory.reshape(B, T, J, self.d_model).mean(dim=1)  # (B, J, d_model)
        
        # Query memory dictionary
        memory_patterns = self.memory_dict(joint_features)  # (B, J, d_model)
    
        # Decoding
     
        output = self.decoder(x, memory, memory_patterns)
       
        # Apply layer norm and final projection
        output = self.final_norm(output)
        output = self.decoder_proj(output).squeeze(-1).reshape(B, T, J)
        
        return output


def create_model(d_model=128, nhead=8, num_layers=3, dim_feedforward=512, device=None):
    """Create and initialize the hand motion prediction model.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of the feedforward network
        device: Device to place the model on
        
    Returns:
        Initialized HandDCTMotionModel
    """
    model = HandDCTMotionModel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )
    
    if device is not None:
        model = model.to(device)
    
    return model