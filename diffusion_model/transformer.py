import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
from .layers import SELayer, TimestepEmbedder, ClusterContinuousEmbedder, CategoricalEmbedder, Mlp
from .conditions import modulate

class ProteinOutLayer(nn.Module):
    def __init__(self, max_n_nodes, hidden_size, aa_type, edge_type, mlp_ratio, num_heads=None):
        super().__init__()
        self.aa_type = aa_type
        self.edge_type = edge_type
        self.max_n_nodes = max_n_nodes
        
        # Separate decoders for amino acids and edges
        self.aa_decoder = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * mlp_ratio,
            out_features=aa_type,
            drop=0
        )
        
        self.edge_decoder = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * mlp_ratio,
            out_features=edge_type,
            drop=0
        )

        # Separate normalization and modulation for aa and edges
        self.norm_aa = nn.LayerNorm(aa_type, elementwise_affine=False)
        self.norm_edge = nn.LayerNorm(edge_type, elementwise_affine=False)
        
        self.adaLN_modulation_aa = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * aa_type, bias=True)
        )
        
        self.adaLN_modulation_edge = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * edge_type, bias=True)
        )

    def forward(self, x, x_in, e_in, c, t, node_mask):
        B, N, _ = x.size()
        
        # Get amino acid predictions
        aa_out = self.aa_decoder(x)  # [B, N, aa_type]
        shift_aa, scale_aa = self.adaLN_modulation_aa(c).chunk(2, dim=1)
        aa_out = modulate(self.norm_aa(aa_out), shift_aa, scale_aa)
        
        # Get edge predictions for each node
        edge_out = self.edge_decoder(x)  # [B, N, edge_type]
        shift_edge, scale_edge = self.adaLN_modulation_edge(c).chunk(2, dim=1)
        edge_out = modulate(self.norm_edge(edge_out), shift_edge, scale_edge)
        
        # For now, we'll create a simple edge matrix by outer product
        # This is a simplified approach - we can improve this later
        edge_out = edge_out.unsqueeze(2) * edge_out.unsqueeze(1)  # [B, N, N, edge_type]
        
        # Add residual connections
        aa_out = x_in + aa_out
        
        # For edges, we need to handle the dense e_in properly
        if e_in.dim() == 4:  # Dense edges [B, N, N, edge_type]
            edge_out = e_in + edge_out
        else:  # Sparse edges - we'll handle this differently
            # For now, just return the dense edge predictions
            pass

        # Handle edge masking
        node_mask_bool = node_mask  # Already bool by standardization
        edge_mask = (~node_mask_bool)[:, :, None] & (~node_mask_bool)[:, None, :]
        diag_mask = (
            torch.eye(N, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .type_as(edge_mask)
        )
        edge_out.masked_fill_(edge_mask[:, :, :, None], 0)
        edge_out.masked_fill_(diag_mask[:, :, :, None], 0)
        edge_out = 1 / 2 * (edge_out + torch.transpose(edge_out, 1, 2))

        return aa_out, edge_out

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
        ydim=3,
        task_type='regression',
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ydim = ydim
        self.Xdim = Xdim
        self.Edim = Edim
        
        # Separate embedders for structure
        # For sparse kNN, we'll embed node features and edge features separately
        self.node_embedder = nn.Linear(Xdim, hidden_size, bias=False)
        self.edge_embedder = nn.Linear(Edim, hidden_size, bias=False)
        
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
        _constant_init(self.out_layer.adaLN_modulation_aa[0], 0)
        _constant_init(self.out_layer.adaLN_modulation_edge[0], 0)

    def forward(self, x, e, p, node_mask, y, t, unconditioned=False, edge_index=None, batch=None):
        """
        Forward pass for protein diffusion model with sparse kNN edges.
        Args:
            x: Node features (amino acids) [B, N, Xdim]
            e: Edge features (distances) [total_edges, Edim] - sparse kNN edges
            p: Sequence features [B, N, Xdim] - same as node features
            node_mask: Node mask [B, N]
            y: Target features (not used in foundation model) [B, ydim]
            t: Timestep [B]
            unconditioned: Whether to use unconditional generation (default: False)
            edge_index: Edge indices [2, total_edges] - for sparse edges
            batch: Batch assignment [total_nodes] - for sparse edges
        """
        # For foundation model training, we don't use conditional features
        force_drop_id = torch.zeros_like(y.sum(-1), dtype=torch.bool)
        
        x_in, p_in, y_in = x, p, y
        bs, n, _ = x.size()
        
        # Convert sparse edges to dense for the OutLayer
        # This is the only place we need dense edges - for the loss function
        dense_e = torch.zeros(bs, n, n, self.Edim, device=x.device)
        if edge_index is not None and batch is not None:
            # Reconstruct dense edge matrix from sparse edges
            for i in range(bs):
                # Get edges for this protein
                protein_node_mask = (batch == i)  # [total_nodes]
                protein_node_indices = torch.where(protein_node_mask)[0]  # Indices of nodes in this protein
                
                if len(protein_node_indices) > 0:
                    start_idx = protein_node_indices[0]
                    end_idx = protein_node_indices[-1] + 1
                    actual_nodes = len(protein_node_indices)
                    
                    # Filter edges where both source and target are in this protein
                    src_in_protein = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
                    dst_in_protein = (edge_index[1] >= start_idx) & (edge_index[1] < end_idx)
                    protein_edge_mask = src_in_protein & dst_in_protein
                    
                    protein_edge_indices = edge_index[:, protein_edge_mask]  # [2, num_edges_in_protein]
                    protein_edge_features = e[protein_edge_mask]  # [num_edges_in_protein, Edim]
                    
                    # Convert to local indices (0 to actual_nodes-1)
                    local_src = protein_edge_indices[0] - start_idx
                    local_dst = protein_edge_indices[1] - start_idx
                    
                    # Fill dense matrix only for actual nodes (the rest will be zero)
                    dense_e[i, local_src, local_dst] = protein_edge_features
        
        e_in = dense_e  # Now dense [B, N, N, Edim]
        
        # Embed node features
        node_emb = self.node_embedder(x)  # [B, N, hidden_size]
        
        # Embed sequence features (same as node features)
        seq_emb = self.node_embedder(p)  # [B, N, hidden_size]
        
        # For now, just use node features without edge aggregation
        # This is a simplified approach - we can add edge aggregation later if needed
        struct_emb = node_emb
        
        # Apply cross-attention between structure and sequence
        struct_emb = self.norm1(struct_emb)
        seq_emb = self.norm2(seq_emb)
        
        # Ensure node_mask is boolean for the ~ operator
        node_mask_bool = node_mask.bool()
        
        # Cross-attention: structure attends to sequence
        struct_attn, _ = self.cross_attention(
            struct_emb, seq_emb, seq_emb,
            key_padding_mask=~node_mask_bool
        )
        struct_emb = struct_emb + struct_attn
        
        # Cross-attention: sequence attends to structure
        seq_attn, _ = self.cross_attention(
            seq_emb, struct_emb, struct_emb,
            key_padding_mask=~node_mask_bool
        )
        seq_emb = seq_emb + seq_attn
        
        # Combine structure and sequence embeddings
        x = struct_emb + seq_emb

        # Time embedding
        print(f"[DEBUG] Transformer - t shape before embedder: {t.shape}", flush=True)
        c1 = self.t_embedder(t)
        print(f"[DEBUG] Transformer - c1 shape after embedder: {c1.shape}", flush=True)
        
        # For foundation model, we don't use target embeddings
        c = c1
        
        for i, block in enumerate(self.encoders):
            x = block(x, c, node_mask)

        # X: B * N * dx, E: B * N * N * de
        X, E = self.out_layer(x, x_in, e_in, c, t, node_mask)
        return utils.PlaceHolder(X=X, E=E).mask(node_mask) 