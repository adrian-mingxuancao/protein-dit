import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj, remove_self_loops
from dataclasses import dataclass
from .graph_utils import PlaceHolder, encode_no_edge, to_dense
from .general_utils import (
    get_project_root,
    setup_paths,
    ensure_dir,
    count_parameters,
    set_seed
)

def create_fully_connected_edges(n_nodes, batch_size):
    """Create a fully connected graph with proper batch handling."""
    edge_index = []
    for b in range(batch_size):
        start_idx = b * n_nodes
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # No self-loops
                    edge_index.append([start_idx + i, start_idx + j])
    return torch.tensor(edge_index, dtype=torch.long).t()

__all__ = [
    'PlaceHolder',
    'encode_no_edge',
    'to_dense',
    'create_fully_connected_edges',
    'get_project_root',
    'setup_paths',
    'ensure_dir',
    'count_parameters',
    'set_seed'
]

@dataclass
class PlaceHolder:
    X: torch.Tensor  # atom features
    E: torch.Tensor  # edge features
    P: torch.Tensor = None  # protein sequence features
    y: torch.Tensor = None  # target features

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # (bs, n, 1)
        e_mask1 = x_mask.unsqueeze(2)             # (bs, n, 1, 1)
        e_mask2 = x_mask.unsqueeze(1)             # (bs, 1, n, 1)

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            if self.P is not None:
                self.P = torch.argmax(self.P, dim=-1)

        self.X = self.X * x_mask
        self.E = self.E * e_mask1 * e_mask2
        if self.P is not None:
            self.P = self.P * x_mask

        assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))

        return self

def encode_no_edge(E):
    """Encode no-edge as a separate edge type."""
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=-1) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E

def to_dense(x, edge_index, edge_attr, batch, max_num_nodes=None):
    """Convert PyG data to dense format."""
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    
    # Handle case where edge_index or edge_attr is None
    if edge_index is None:
        # Create a fully connected graph
        edge_index = create_fully_connected_edges(X.size(1), X.size(0))
        edge_attr = None
    
    # Remove self loops if edge_index exists
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    
    if max_num_nodes is None:
        max_num_nodes = X.size(1)
    
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask
