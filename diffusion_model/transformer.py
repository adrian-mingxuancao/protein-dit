import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
from .layers import SELayer, TimestepEmbedder, ClusterContinuousEmbedder, CategoricalEmbedder, Mlp
from .conditions import modulate

class ProteinOutLayer(nn.Module):
    def __init__(self, max_n_nodes, hidden_size, aa_type, edge_type, seq_type, mlp_ratio, num_heads=None):
        super().__init__()
        self.aa_type = aa_type
        self.edge_type = edge_type
        self.seq_type = seq_type
        final_size = aa_type + max_n_nodes * edge_type + seq_type
        
        # Separate decoders for structure and sequence
        self.struct_decoder = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * mlp_ratio,
            out_features=aa_type + max_n_nodes * edge_type,
            drop=0
        )
        
        self.seq_decoder = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * mlp_ratio,
            out_features=seq_type,
            drop=0
        )

        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x, x_in, e_in, p_in, c, t, node_mask):
        # Get structure and sequence predictions
        struct_out = self.struct_decoder(x)
        seq_out = self.seq_decoder(x)
        
        B, N, D = struct_out.size()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        # Apply modulation to both outputs
        struct_out = modulate(self.norm_final(struct_out), shift[:, :, :struct_out.size(-1)], scale[:, :, :struct_out.size(-1)])
        seq_out = modulate(self.norm_final(seq_out), shift[:, :, struct_out.size(-1):], scale[:, :, struct_out.size(-1):])
        
        # Split structure outputs for amino acids and edges
        aa_out = struct_out[:, :, :self.aa_type]
        edge_out = struct_out[:, :, self.aa_type:].reshape(B, N, N, self.edge_type)
        
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

        return aa_out, edge_out, seq_out

class ProteinDenoiser(nn.Module):
    def __init__(
        self,
        max_n_nodes,
        hidden_size=384,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        drop_condition=0.1,
        Xdim=20,  # 20 amino acids
        Edim=5,   # 5 distance bins
        Pdim=20,  # 20 amino acids for sequence
        ydim=3,
        task_type='regression',
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ydim = ydim
        
        # Separate embedders for structure and sequence
        self.struct_embedder = nn.Linear(Xdim + max_n_nodes * Edim, hidden_size, bias=False)
        self.seq_embedder = nn.Linear(Pdim, hidden_size, bias=False)
        
        # Cross-attention between structure and sequence
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # Layer norm for cross-attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedding_list = torch.nn.ModuleList()

        self.y_embedding_list.append(ClusterContinuousEmbedder(2, hidden_size, drop_condition))
        for i in range(ydim - 2):
            if task_type == 'regression':
                self.y_embedding_list.append(ClusterContinuousEmbedder(1, hidden_size, drop_condition))
            else:
                self.y_embedding_list.append(CategoricalEmbedder(2, hidden_size, drop_condition))

        self.encoders = nn.ModuleList(
            [
                SELayer(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.out_layer = ProteinOutLayer(
            max_n_nodes=max_n_nodes,
            hidden_size=hidden_size,
            aa_type=Xdim,
            edge_type=Edim,
            seq_type=Pdim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)

        for block in self.encoders:
            _constant_init(block.adaLN_modulation[0], 0)
        _constant_init(self.out_layer.adaLN_modulation[0], 0)

    def forward(self, x, e, p, node_mask, y, t, unconditioned=False):
        """
        Forward pass for protein diffusion model.
        Args:
            x: Node features (amino acids) [B, N, Xdim]
            e: Edge features (distances) [B, N, N, Edim]
            p: Sequence features [B, N, Pdim]
            node_mask: Node mask [B, N]
            y: Target features (not used in foundation model) [B, ydim]
            t: Timestep [B]
            unconditioned: Whether to use unconditional generation (default: False)
        """
        # For foundation model training, we don't use conditional features
        force_drop_id = torch.zeros_like(y.sum(-1), dtype=torch.bool)
        
        x_in, e_in, p_in, y_in = x, e, p, y
        bs, n, _ = x.size()
        
        # Embed structure and sequence separately
        struct_emb = self.struct_embedder(torch.cat([x, e.reshape(bs, n, -1)], dim=-1))
        seq_emb = self.seq_embedder(p)
        
        # Apply cross-attention between structure and sequence
        struct_emb = self.norm1(struct_emb)
        seq_emb = self.norm2(seq_emb)
        
        # Cross-attention: structure attends to sequence
        struct_attn, _ = self.cross_attention(
            struct_emb, seq_emb, seq_emb,
            key_padding_mask=~node_mask
        )
        struct_emb = struct_emb + struct_attn
        
        # Cross-attention: sequence attends to structure
        seq_attn, _ = self.cross_attention(
            seq_emb, struct_emb, struct_emb,
            key_padding_mask=~node_mask
        )
        seq_emb = seq_emb + seq_attn
        
        # Combine structure and sequence embeddings
        x = struct_emb + seq_emb

        # Time embedding
        c1 = self.t_embedder(t)
        
        # For foundation model, we don't use target embeddings
        c = c1
        
        for i, block in enumerate(self.encoders):
            x = block(x, c, node_mask)

        # X: B * N * dx, E: B * N * N * de, P: B * N * dp
        X, E, P = self.out_layer(x, x_in, e_in, p_in, c, t, node_mask)
        return utils.PlaceHolder(X=X, E=E, y=P).mask(node_mask) 