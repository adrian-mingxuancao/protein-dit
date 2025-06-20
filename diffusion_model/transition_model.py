import torch
import torch.nn.functional as F
import gc
from torch_geometric.utils import to_dense_batch, to_dense_adj, remove_self_loops
from torch_geometric.nn import radius_graph, knn_graph
from dataclasses import dataclass

class ProteinTransitionModel:
    def __init__(self, x_marginals, e_marginals, p_marginals, xe_conditions, ex_conditions, y_classes, n_nodes):
        """
        Initialize transition model that captures all state dependencies:
        - Node → Node transitions (amino acid types)
        - Edge → Node transitions (structure to sequence)
        - Edge → Edge transitions (structural dependencies)
        - Node → Edge transitions (sequence to structure)
        
        Args:
            x_marginals: Marginal distribution of amino acid types
            e_marginals: Marginal distribution of edge types (distance bins)
            p_marginals: Marginal distribution of protein sequence positions (kept for future use)
            xe_conditions: Conditional distribution of edge types given amino acid types [X, E]
            ex_conditions: Conditional distribution of amino acid types given edge types [E, X]
            y_classes: Number of global feature classes
            n_nodes: Maximum number of nodes in a graph
        """
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.P_classes = len(p_marginals)  # Kept for future use
        self.y_classes = y_classes
        self.n_nodes = n_nodes
        
        # Store marginals and conditionals
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals
        self.p_marginals = p_marginals  # Kept for future use
        self.xe_conditions = xe_conditions
        self.ex_conditions = ex_conditions
        
        # Create transition matrices for each dependency
        # Node → Node transitions (amino acids)
        self.Q_xx = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)  # [1, X, X]
        
        # Edge → Node transitions [1, E, X]
        self.Q_ex = ex_conditions.unsqueeze(0)  # [1, E, X] = [1, 5, 20]
        
        # Edge → Edge transitions [1, E, E]
        self.Q_ee = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        
        # Node → Edge transitions [1, X, E]
        self.Q_xe = xe_conditions.unsqueeze(0)  # [1, X, E] = [1, 20, 5]
        
        # Protein sequence transitions [1, P, P] - commented out for now
        # self.Q_pp = p_marginals.unsqueeze(0).expand(self.P_classes, -1).unsqueeze(0)

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

    def _get_structure_transitions(self, alpha_bar, device, batch_size, edge_index=None, edge_method='radius'):
        """Get structure transition probabilities."""
        if edge_method == 'dense':
            # For dense batched data, create transition matrices for each protein
            # Use actual protein size (2699+ nodes) instead of Graph-DiT's 800
            n_nodes = 3000  # Max nodes for proteins (larger than your 2699)
            transitions = []
            for i in range(batch_size):
                # Create identity matrix for each protein
                Q_struct = torch.eye(n_nodes, device=device)
                
                # Apply noise schedule
                Q_struct = alpha_bar[i] * Q_struct + (1 - alpha_bar[i]) * torch.ones_like(Q_struct) / n_nodes
                
                transitions.append(Q_struct)
            
            return transitions
        elif edge_method == 'knn':
            # For kNN, use sparse matrices for memory efficiency
            edge_indices = edge_index  # [2, num_edges]
            num_edges = edge_indices.size(1)
            
            # Get unique nodes from edge indices
            unique_nodes = torch.unique(edge_indices)
            n_nodes = len(unique_nodes)
            
            print(f"[DEBUG] kNN transition: {n_nodes} nodes, {num_edges} edges", flush=True)
            
            # Create sparse transition matrix for memory efficiency
            print(f"[DEBUG] Creating sparse transition matrix for {n_nodes} nodes", flush=True)
            
            # Create diagonal elements (self-transitions)
            diag_indices = torch.arange(n_nodes, device=device)
            diag_indices = torch.stack([diag_indices, diag_indices], dim=0)
            diag_values = alpha_bar * torch.ones(n_nodes, device=device)
                        
            # Create off-diagonal elements (kNN transitions)
            # For simplicity, assume uniform transitions to neighbors
            off_diag_values = (1 - alpha_bar) / max(num_edges, 1) * torch.ones(num_edges, device=device)
                        
            # Combine indices and values
            all_indices = torch.cat([diag_indices, edge_indices], dim=1)
            all_values = torch.cat([diag_values, off_diag_values])
            
            # Create sparse tensor
            Q_struct = torch.sparse_coo_tensor(
                all_indices, 
                all_values, 
                size=(n_nodes, n_nodes),
                device=device,
                dtype=torch.float32  # Force FP32 for sparse operations
            ).coalesce()  # Remove duplicates and sort
            
            print(f"[DEBUG] Created sparse matrix: {Q_struct.shape}, {Q_struct._nnz()} non-zero elements", flush=True)
            
            return [Q_struct]
        else:
            # For radius method, use dense matrices
            n_nodes = 3000  # Max nodes for proteins
            Q_struct = torch.eye(n_nodes, device=device)
            Q_struct = alpha_bar * Q_struct + (1 - alpha_bar) * torch.ones_like(Q_struct) / n_nodes
            return [Q_struct]

    """
    def _get_sequence_transitions(self, alpha_bar, device, batch_size):
        # Get sequence transition probabilities.
        
        Args:
            alpha_bar: Noise level at timestep t
            device: Device to place tensors on
            batch_size: Batch size
            
        Returns:
            Q_p: Transition probabilities for each position [B, n, 20]
        
        # Create base probabilities for each amino acid
        p_marginals = self.p_marginals.to(device)  # [20]
        
        # Create identity for current state
        I = torch.eye(self.P_classes, device=device)  # [20, 20]
        
        # Compute transition probabilities for each position
        # This is a 20-dimensional vector (probabilities for each amino acid)
        Q_p = alpha_bar * I + (1 - alpha_bar) * p_marginals.unsqueeze(0)  # [20, 20]
        
        # Repeat for batch and number of nodes
        Q_p = Q_p.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_nodes, 1, 1)  # [B, n, 20, 20]
        
        return Q_p  # [B, n, 20, 20] - much smaller than current (20n)x(20n) matrix
    """

    def get_Qt_bar(self, alpha_bar_t, device, batch_size=1, edge_index=None, edge_method='radius'):
        """Get transition matrix for timestep t."""
        # Reshape alpha_bar for broadcasting
        alpha_bar = alpha_bar_t.view(-1, 1, 1)
        
        # Calculate dimensions
        n_edges = self.n_nodes * self.E_classes
        
        # Skip chunking for single proteins to avoid multiple transition matrix calls
        if batch_size == 1:
            print(f"[DEBUG] Single protein - skipping chunking for {self.n_nodes} nodes", flush=True)
            # Get structure transitions directly
            Q_struct = self._get_structure_transitions(
                alpha_bar, 
                device, 
                batch_size,
                edge_index=edge_index,
                edge_method=edge_method
            )
            return [Q_struct]  # Return as list to maintain compatibility
        
        # Use very small chunks to reduce memory usage (only for batch_size > 1)
        chunk_size = min(2, batch_size)  # Process only 2 samples at a time
        results = []
        
        for i in range(0, batch_size, chunk_size):
            current_chunk_size = min(chunk_size, batch_size - i)
            chunk_alpha_bar = alpha_bar[i:i+current_chunk_size]
            
            # Get structure transitions for this chunk
            Q_struct = self._get_structure_transitions(
                chunk_alpha_bar, 
                device, 
                current_chunk_size,
                edge_index=edge_index,
                edge_method=edge_method
            )
            
            # Store structure matrix
            results.append(Q_struct)
            
            # Clear memory
            del Q_struct
            torch.cuda.empty_cache()
            gc.collect()
        
        return results

    def get_conditional_transition(self, x, e, p, t, device):
        """Get conditional transition probabilities."""
        # Get transition matrices
        transitions = self.get_Qt_bar(t, device, batch_size=x.size(0))
        
        # Process each chunk
        x_probs_list = []
        e_probs_list = []
        
        for Q_struct in transitions:
            # Process structure transitions
            x_flat = x.reshape(x.size(0), -1)  # [B, X]
            e_flat = e.reshape(e.size(0), -1)  # [B, n*E]
            struct_input = torch.cat([x_flat, e_flat], dim=1)  # [B, X+n*E]
            struct_probs = struct_input @ Q_struct  # [B, X+n*E]
            
            # Split structure probabilities
            x_probs = struct_probs[:, :self.X_classes]
            e_probs = struct_probs[:, self.X_classes:]
            e_probs = e_probs.reshape(-1, self.n_nodes, self.n_nodes, self.E_classes)
            
            x_probs_list.append(x_probs)
            e_probs_list.append(e_probs)
            
            # Clear memory
            del struct_input, struct_probs
            torch.cuda.empty_cache()
            gc.collect()
        
        # Combine results from all chunks
        x_probs = torch.cat(x_probs_list, dim=0)
        e_probs = torch.cat(e_probs_list, dim=0)
        
        return x_probs, e_probs, None  # Return None for p_probs to maintain compatibility

    def _get_amino_acid_transitions(self, alpha_bar, device, batch_size):
        """Get amino acid transition matrix."""
        # Create identity matrix
        I_x = torch.eye(self.X_classes, device=device).unsqueeze(0)  # [1, X, X]
        
        # Compute Qt_bar for amino acids
        Q_x = alpha_bar * I_x + (1 - alpha_bar) * self.Q_xx.to(device)  # [bs, X, X]
        
        return Q_x

    def _get_edge_transitions(self, alpha_bar, device, batch_size):
        """Get edge transition matrix."""
        # Create identity matrix
        I_e = torch.eye(self.E_classes, device=device).unsqueeze(0)  # [1, E, E]
        
        # Compute Qt_bar for edges
        Q_e = alpha_bar * I_e + (1 - alpha_bar) * self.Q_ee.to(device)  # [bs, E, E]
        
        # Expand edge transitions
        Q_e = Q_e.repeat(1, self.n_nodes, self.n_nodes)  # [bs, n*E, n*E]
        
        return Q_e

    def _get_protein_transitions(self, alpha_bar, device, batch_size):
        """Get protein sequence transition matrix."""
        # Create identity matrix
        I_p = torch.eye(self.P_classes, device=device).unsqueeze(0)  # [1, P, P]
        
        # Compute Qt_bar for proteins
        Q_p = alpha_bar * I_p + (1 - alpha_bar) * self.Q_pp.to(device)  # [bs, P, P]
        
        # Expand protein transitions
        Q_p = Q_p.repeat(1, self.n_nodes, 1)  # [bs, n*P, P]
        
        return Q_p 

def create_fully_connected_edges(n_nodes, batch_size=1, coords=None, distance_threshold=10.0, max_num_neighbors=32, edge_method='radius', k=16):
    """
    Create edges based on either distance threshold (radius_graph) or k-nearest neighbors (knn_graph).
    Args:
        n_nodes: Number of nodes
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
        # If coordinates provided, use selected method
        if batch_size > 1:
            batch = torch.arange(batch_size, device=coords.device).repeat_interleave(n_nodes)
            coords = coords.view(-1, 3)  # Reshape coords to [batch_size * n_nodes, 3]
        else:
            batch = None
        if edge_method == 'knn':
            edge_index = knn_graph(
                coords,
                k=k,
                batch=batch,
                loop=False
            )
        else:  # default to radius_graph
            edge_index = radius_graph(
                coords,
                r=distance_threshold,
                batch=batch,
                max_num_neighbors=max_num_neighbors
            )
        # If batch_size > 1, reshape edge_index back to original batch format
        if batch_size > 1:
            edge_index = edge_index.view(2, batch_size, -1)
            edge_index = edge_index.permute(1, 0, 2).reshape(2, -1)
    return edge_index 