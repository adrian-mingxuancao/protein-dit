import torch
import torch.nn as nn
import torch.nn.functional as F
from .conditions import modulate
from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, t):
        return self.mlp(t)


class ClusterContinuousEmbedder(nn.Module):
    def __init__(self, input_dims, hidden_size, drop_condition):
        super().__init__()
        self.drop_condition = drop_condition
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x, training, force_drop_id, t):
        if training and self.drop_condition > 0:
            drop_mask = torch.bernoulli(torch.ones_like(force_drop_id) * self.drop_condition).bool()
            force_drop_id = force_drop_id | drop_mask
        x = torch.where(force_drop_id.unsqueeze(-1), torch.zeros_like(x), x)
        return self.mlp(x)


class CategoricalEmbedder(nn.Module):
    def __init__(self, input_dims, hidden_size, drop_condition):
        super().__init__()
        self.drop_condition = drop_condition
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x, training, force_drop_id, t):
        if training and self.drop_condition > 0:
            drop_mask = torch.bernoulli(torch.ones_like(force_drop_id) * self.drop_condition).bool()
            force_drop_id = force_drop_id | drop_mask
        x = torch.where(force_drop_id.unsqueeze(-1), torch.zeros_like(x), x)
        return self.mlp(x)


def modulate(x, shift, scale):
    """Modulate the input tensor with shift and scale parameters."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class SELayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = ProteinAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = ProteinMLP(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, node_mask):
        print("\nSELayer shapes:")
        print(f"x: {x.shape}")
        print(f"c: {c.shape}")
        print(f"node_mask: {node_mask.shape}")
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        print(f"shift_msa: {shift_msa.shape}")
        print(f"gate_msa: {gate_msa.shape}")
        
        # First attention path
        attn_out = self.attn(x, node_mask=node_mask)
        print(f"attn_out: {attn_out.shape}")
        norm_attn = self.norm1(attn_out)
        print(f"norm_attn: {norm_attn.shape}")
        
        # Apply modulation using the modulate function
        modulated_attn = modulate(norm_attn, shift_msa, scale_msa)
        print(f"modulated_attn: {modulated_attn.shape}")
        x = x + gate_msa.unsqueeze(1) * modulated_attn
        
        # Second MLP path
        mlp_out = self.mlp(x)
        print(f"mlp_out: {mlp_out.shape}")
        norm_mlp = self.norm2(mlp_out)
        print(f"norm_mlp: {norm_mlp.shape}")
        
        # Apply modulation using the modulate function
        modulated_mlp = modulate(norm_mlp, shift_mlp, scale_mlp)
        print(f"modulated_mlp: {modulated_mlp.shape}")
        x = x + gate_mlp.unsqueeze(1) * modulated_mlp
        
        return x


class ProteinAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, node_mask):
        print("\nProteinAttention shapes:")
        print(f"x: {x.shape}")
        print(f"node_mask: {node_mask.shape}")
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        print(f"q: {q.shape}")
        print(f"k: {k.shape}")
        print(f"v: {v.shape}")

        attn = (q @ k.transpose(-2, -1)) * self.scale
        print(f"attn before mask: {attn.shape}")
        
        # Apply node mask - expand to match attention dimensions
        node_mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        print(f"expanded node_mask: {node_mask.shape}")
        attn = attn.masked_fill(~node_mask, float('-inf'))
        print(f"attn after mask: {attn.shape}")
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        print(f"x after attention: {x.shape}")
        x = self.proj(x)
        x = self.proj_drop(x)
        print(f"x final: {x.shape}")
        return x


class ProteinMLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OutLayer(nn.Module):
    def __init__(self, max_n_nodes, hidden_size, aa_type, edge_type, seq_type, mlp_ratio=4.0, num_heads=4):
        super().__init__()
        self.max_n_nodes = max_n_nodes
        self.hidden_size = hidden_size
        self.aa_type = aa_type
        self.edge_type = edge_type
        self.seq_type = seq_type
        self.mlp_ratio = float(mlp_ratio)  # Ensure mlp_ratio is a float
        self.num_heads = num_heads

        # Calculate final size for linear layer
        final_size = aa_type + max_n_nodes * edge_type + seq_type

        # Final MLP for output
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * self.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * self.mlp_ratio), final_size)
        )

        # Add back normalization and modulation
        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x, x_in, e_in, p_in, c, t, node_mask):
        # Process through final MLP
        x_all = self.final_mlp(x)
        B, N, D = x_all.size()
        
        # Apply modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_all = modulate(self.norm_final(x_all), shift, scale)
        
        # Split outputs for amino acids, edges, and sequence
        aa_out = x_all[:, :, :self.aa_type]
        edge_out = x_all[:, :, self.aa_type:self.aa_type + N * self.edge_type].reshape(B, N, N, self.edge_type)
        seq_out = x_all[:, :, self.aa_type + N * self.edge_type:]
        
        # Add residual connections
        aa_out = x_in + aa_out
        edge_out = e_in + edge_out
        seq_out = p_in + seq_out

        # Handle edge masking
        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = (
            torch.eye(N, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .type_as(edge_mask)
        )
        edge_out.masked_fill_(edge_mask[:, :, :, None], 0)
        edge_out.masked_fill_(diag_mask[:, :, :, None], 0)
        edge_out = 1 / 2 * (edge_out + torch.transpose(edge_out, 1, 2))

        return aa_out, edge_out, seq_out, None


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if isinstance(hidden_features, float):
            hidden_features = int(hidden_features)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x 