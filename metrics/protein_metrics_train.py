import torch
import torch.nn.functional as F
from metrics.protein_metrics_sampling import SumExceptBatchKL, NLL
import time

class TrainLossDiscrete:
    def __init__(self, lambda_train):
        self.lambda_train = lambda_train
        self.train_X_kl = SumExceptBatchKL()
        self.train_E_kl = SumExceptBatchKL()
        self.train_X_logp = NLL()
        self.train_E_logp = NLL()

    def __call__(self, masked_pred_X, masked_pred_E, true_X, true_E, edge_index=None, batch=None, log=True):
        """Compute training loss for protein design with sparse edges.
        
        Args:
            masked_pred_X: Predicted node features (amino acids) [B, N, Xdim]
            masked_pred_E: Predicted edge features (contacts) [B, N, N, Edim] - dense predictions
            true_X: Ground truth node features [B, N, Xdim]
            true_E: Ground truth edge features [total_edges, Edim] - sparse kNN edges
            edge_index: Edge indices [2, total_edges] - for sparse edges
            batch: Batch indices [total_nodes] - for sparse edges
            log: Whether to log metrics
        """
        # Compute KL divergence for nodes (amino acids)
        X_kl = self.train_X_kl(masked_pred_X, true_X)
        
        # Compute log probability
        X_logp = self.train_X_logp(masked_pred_X, true_X)
        
        # For edges, we need to extract predictions for actual edges from dense predictions
        if edge_index is not None and batch is not None:
            # Extract edge predictions for actual edges
            batch_size = masked_pred_E.size(0)
            max_nodes = masked_pred_E.size(1)
            
            # Create dense edge matrix from sparse edges for comparison
            dense_E = torch.zeros(batch_size, max_nodes, max_nodes, masked_pred_E.size(-1), device=true_X.device)
            
            # Reconstruct dense edge matrix from sparse edges
            for i in range(batch_size):
                # Get edges for this protein
                protein_node_mask = (batch == i)  # [total_nodes]
                protein_node_indices = torch.where(protein_node_mask)[0]  # Indices of nodes in this protein
                
                # Get edges that belong to this protein
                protein_edge_mask = torch.isin(edge_index[0], protein_node_indices)  # [total_edges]
                protein_edges = edge_index[:, protein_edge_mask]  # [2, protein_edges]
                protein_edge_features = true_E[protein_edge_mask]  # [protein_edges, Edim]
                
                # Convert global indices to local indices
                local_edge_index = torch.zeros_like(protein_edges)
                for j, global_idx in enumerate(protein_node_indices):
                    local_edge_index[0][protein_edges[0] == global_idx] = j
                    local_edge_index[1][protein_edges[1] == global_idx] = j
                
                # Fill dense matrix
                dense_E[i, local_edge_index[0], local_edge_index[1]] = protein_edge_features
            
            # Compute edge loss on dense matrices
            E_kl = self.train_E_kl(masked_pred_E, dense_E)
            E_logp = self.train_E_logp(masked_pred_E, dense_E)
        else:
            # Fallback to dense comparison if no sparse info
            E_kl = self.train_E_kl(masked_pred_E, true_E)
            E_logp = self.train_E_logp(masked_pred_E, true_E)
        
        # Combine losses with weights (using indices like original Graph-DiT)
        loss = (
            self.lambda_train[0] * X_kl +
            self.lambda_train[1] * E_kl
        )
        
        if log:
            print(f"Train X KL: {X_kl:.4f} -- Train E KL: {E_kl:.4f}")
            print(f"Train X logp: {X_logp:.4f} -- Train E logp: {E_logp:.4f}")
            print(f"Lambda weights: X={self.lambda_train[0]}, E={self.lambda_train[1]}")
            print(f"Total loss: {loss:.4f}")
        
        return loss

    def reset(self):
        self.train_X_kl.reset()
        self.train_E_kl.reset()
        self.train_X_logp.reset()
        self.train_E_logp.reset() 

    def log_epoch_metrics(self, current_epoch, start_epoch_time, log=True):
        """Log epoch metrics like original Graph-DiT."""
        if log:
            print(f"Epoch {current_epoch} finished: "
                  f"X_KL: {self.train_X_kl.compute():.4f} -- E_KL: {self.train_E_kl.compute():.4f} "
                  f"X_logp: {self.train_X_logp.compute():.4f} -- E_logp: {self.train_E_logp.compute():.4f} "
                  f"-- Time taken {time.time() - start_epoch_time:.1f}s") 