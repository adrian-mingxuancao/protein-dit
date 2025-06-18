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
        if edge_method == 'knn':
            # Get edge indices
            edge_indices = edge_index  # Already in [2, num_edges] format
            num_edges = edge_indices.size(1)
            print(f"\nDEBUG: Starting _get_structure_transitions")
            print(f"DEBUG: edge_indices shape: {edge_indices.shape}")
            print(f"DEBUG: batch_size: {batch_size}")
            print(f"DEBUG: num_edges: {num_edges}")
            print(f"DEBUG: Current GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            
            # Process edges in smaller chunks to avoid OOM
            chunk_size = 1000  # Process 1000 edges at a time
            all_shared_edges = []
            total_shared = 0
            
            for i in range(0, num_edges, chunk_size):
                end_idx = min(i + chunk_size, num_edges)
                chunk_src = edge_indices[0, i:end_idx]
                chunk_dst = edge_indices[1, i:end_idx]
                print(f"\nDEBUG: Processing chunk {i//chunk_size + 1}")
                print(f"DEBUG: Current GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                
                # For each edge in the chunk, find edges that share nodes
                chunk_shared = 0
                for j in range(len(chunk_src)):
                    src = chunk_src[j]
                    dst = chunk_dst[j]
                    
                    # Find edges that share source
                    shared_src = (edge_indices[0] == src) | (edge_indices[1] == src)
                    # Find edges that share destination
                    shared_dst = (edge_indices[0] == dst) | (edge_indices[1] == dst)
                    # Combine and remove self
                    shared = shared_src | shared_dst
                    shared[i + j] = False
                    
                    # Get indices of shared edges
                    shared_indices = torch.nonzero(shared).squeeze(-1)
                    
                    if len(shared_indices) > 0:
                        chunk_shared += len(shared_indices)
                        # Only print debug for the first 5 edges in the first chunk
                        if i == 0 and j < 5:
                            print(f"DEBUG: Edge {i+j} ({src.item()}->{dst.item()}) shares nodes with {len(shared_indices)} other edges")
                            # Print some example shared edges
                            if len(shared_indices) > 0:
                                example_shared = shared_indices[:3]  # First 3 shared edges
                                for idx in example_shared:
                                    s = edge_indices[0, idx].item()
                                    d = edge_indices[1, idx].item()
                                    print(f"  - Shared edge {idx}: {s}->{d}")
                        
                        # Create indices for this edge's transitions
                        indices = torch.stack([
                            torch.zeros_like(shared_indices),  # batch dimension
                            torch.full_like(shared_indices, i + j),  # source edge
                            shared_indices  # target edge
                        ])
                        
                        # Process this chunk immediately to avoid storing all indices
                        values = torch.full((indices.size(1),), alpha_bar[0].item(), device=device)
                        Q_chunk = torch.sparse_coo_tensor(
                            indices=indices,
                            values=values,
                            size=(batch_size, num_edges, num_edges)
                        )
                        
                        # Add to running sum
                        if len(all_shared_edges) == 0:
                            all_shared_edges = Q_chunk
                        else:
                            all_shared_edges = all_shared_edges + Q_chunk
                        
                        # Clear memory
                        del Q_chunk
                        torch.cuda.empty_cache()
                
                total_shared += chunk_shared
                print(f"DEBUG: Chunk {i//chunk_size + 1} shared edges: {chunk_shared}")
                print(f"DEBUG: Average shared edges per edge in chunk: {chunk_shared/len(chunk_src):.2f}")
                
                # Clear memory
                del chunk_src, chunk_dst
                torch.cuda.empty_cache()
            
            if len(all_shared_edges) == 0:
                # If no shared edges, return empty sparse tensor
                return torch.sparse_coo_tensor(
                    indices=torch.zeros((2, 0), device=device, dtype=torch.long),
                    values=torch.zeros(0, device=device),
                    size=(batch_size, num_edges, num_edges)
                )
            
            print(f"\nDEBUG: Total shared edges: {total_shared}")
            print(f"DEBUG: Average shared edges per edge: {total_shared/num_edges:.2f}")
            print(f"DEBUG: Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            return all_shared_edges
        
        else:
            # Memory-efficient dense approach for radius_graph
            # Instead of creating full matrix, compute transitions in chunks
            chunk_size = 1000  # Adjust based on available memory
            Q_struct = torch.zeros(batch_size, self.n_nodes * (self.X_classes + self.E_classes), 
                                 self.n_nodes * (self.X_classes + self.E_classes), device=device)
            
            total_size = self.n_nodes * (self.X_classes + self.E_classes)
            for i in range(0, total_size, chunk_size):
                end_i = min(i + chunk_size, total_size)
                for j in range(0, total_size, chunk_size):
                    end_j = min(j + chunk_size, total_size)
                    # Compute transitions for this chunk
                    Q_struct[:, i:end_i, j:end_j] = alpha_bar * torch.eye(
                        end_i - i, device=device
                    ).unsqueeze(0).expand(batch_size, -1, -1)
            
            return Q_struct

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
        
        # Use very small chunks to reduce memory usage
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