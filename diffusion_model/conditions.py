import torch
import torch.nn as nn
import torch.nn.functional as F

def modulate(x, shift, scale):
    """Modulate the input tensor with shift and scale parameters.
    
    Args:
        x: Input tensor of shape [B, N, D]
        shift: Shift tensor of shape [B, D]
        scale: Scale tensor of shape [B, D]
    
    Returns:
        Modulated tensor of shape [B, N, D]
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ProteinConditionalEmbedding(nn.Module):
    def __init__(self, hidden_size, num_conditions=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_conditions = num_conditions
        
        # Embedding for each condition type
        self.condition_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_conditions)
        ])
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * num_conditions, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, conditions):
        # conditions: [B, num_conditions]
        B = conditions.shape[0]
        
        # Process each condition
        embeddings = []
        for i in range(self.num_conditions):
            cond = conditions[:, i:i+1]  # [B, 1]
            emb = self.condition_embeddings[i](cond)  # [B, hidden_size]
            embeddings.append(emb)
        
        # Concatenate and project
        x = torch.cat(embeddings, dim=-1)  # [B, hidden_size * num_conditions]
        x = self.proj(x)  # [B, hidden_size]
        
        return x


class ProteinSequenceCondition(nn.Module):
    def __init__(self, hidden_size, max_seq_len=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Sequence condition processing
        self.seq_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, seq_condition, node_mask):
        # seq_condition: [B, N, hidden_size]
        B, N, _ = seq_condition.shape
        
        # Add positional embeddings
        pos_emb = self.pos_embed[:, :N, :]
        x = seq_condition + pos_emb
        
        # Process sequence
        x = self.seq_encoder(x)
        
        # Apply masking
        x = x.masked_fill(~node_mask.unsqueeze(-1), 0)
        
        return x


class ProteinStructureCondition(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Structure condition processing
        self.struct_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, struct_condition, node_mask):
        # struct_condition: [B, N, N, hidden_size]
        B, N, _, _ = struct_condition.shape
        
        # Process structure
        x = self.struct_encoder(struct_condition)
        
        # Apply masking
        x = x.masked_fill(~node_mask.unsqueeze(-1).unsqueeze(-1), 0)
        
        return x 