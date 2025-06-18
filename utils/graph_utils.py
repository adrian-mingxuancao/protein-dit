import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj, remove_self_loops
from torch_geometric.nn import radius_graph, knn_graph
from dataclasses import dataclass

@dataclass
class PlaceHolder:
    X: torch.Tensor
    E: torch.Tensor
    y: torch.Tensor

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # (bs, n, 1)
        e_mask1 = x_mask.unsqueeze(2)             # (bs, n, 1, 1)
        e_mask2 = x_mask.unsqueeze(1)             # (bs, 1, n, 1)

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

        self.X = self.X * x_mask
        self.E = self.E * e_mask1 * e_mask2

        assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))

        return self

def create_fully_connected_edges(n_nodes, batch_size=1, coords=None, distance_threshold=10.0, max_num_neighbors=32, edge_method='radius', k=16):
    """
    Create edges based on either distance threshold (radius_graph) or k-nearest neighbors (knn_graph).
    For kNN: Creates a symmetric graph where if A is k-nearest to B, then B is connected to A.
    Args:
        n_nodes: Number of nodes per protein
        batch_size: Batch size
        coords: Node coordinates (optional)
        distance_threshold: Distance threshold for radius_graph
        max_num_neighbors: Max neighbors for radius_graph
        edge_method: 'radius' (default) or 'knn' to select edge construction method
        k: Number of neighbors for knn_graph (default 16)
    Returns:
        edge_index: Edge indices
    """
    if coords is None:
        # If no coordinates provided, create fully connected graph
        edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # No self-loops
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    else:
        if batch_size > 1:
            # Process each protein separately
            all_edge_indices = []
            for i in range(batch_size):
                start_idx = i * n_nodes
                end_idx = (i + 1) * n_nodes
                protein_coords = coords[start_idx:end_idx]
                
                if edge_method == 'knn':
                    # Create kNN for this protein
                    protein_edges = knn_graph(
                        protein_coords,
                        k=k,
                        loop=False
                    )
                    # Make it symmetric by adding reverse edges
                    protein_edges = torch.cat([protein_edges, protein_edges.flip(0)], dim=1)
                    # Remove duplicates to keep it sparse
                    protein_edges = torch.unique(protein_edges, dim=1)
                    # Add offset to node indices
                    protein_edges = protein_edges + start_idx
                else:  # radius_graph
                    protein_edges = radius_graph(
                        protein_coords,
                        r=distance_threshold,
                        max_num_neighbors=max_num_neighbors
                    )
                    # Add offset to node indices
                    protein_edges = protein_edges + start_idx
                
                all_edge_indices.append(protein_edges)
    
            # Combine all edge indices
            edge_index = torch.cat(all_edge_indices, dim=1)
        else:
            # Single protein case
            if edge_method == 'knn':
                edge_index = knn_graph(
                    coords,
                    k=k,
                    loop=False
                )
                # Make it symmetric by adding reverse edges
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                # Remove duplicates to keep it sparse
                edge_index = torch.unique(edge_index, dim=1)
            else:  # radius_graph
                edge_index = radius_graph(
                    coords,
                    r=distance_threshold,
                    max_num_neighbors=max_num_neighbors
                )
    
    return edge_index

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
    """Convert PyG data to dense format.
    For kNN: Keeps edge matrix sparse for memory efficiency.
    For radius: Converts to dense format.
    """
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if max_num_nodes is None:
        max_num_nodes = X.size(1)
    
    # Check if this is a kNN graph by looking at edge density
    num_edges = edge_index.size(1)
    num_possible_edges = batch_size * max_num_nodes * max_num_nodes
    edge_density = num_edges / num_possible_edges
    
    # If edge density is low (kNN case), keep it sparse
    if edge_density < 0.1:  # kNN typically has much lower density
        # Create sparse edge matrix
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
        E = torch.sparse_coo_tensor(
            indices=edge_index,
            values=edge_attr,
            size=(batch_size, max_num_nodes, max_num_nodes)
        )
    else:
        # For dense case (radius), use dense matrix
        E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask 