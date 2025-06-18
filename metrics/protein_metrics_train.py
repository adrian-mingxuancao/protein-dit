import torch
import torch.nn.functional as F
from protein_dit.metrics.protein_metrics_sampling import SumExceptBatchMetric

class TrainLossDiscrete:
    def __init__(self, lambda_train):
        self.lambda_train = lambda_train
        self.train_X_kl = SumExceptBatchMetric()
        self.train_E_kl = SumExceptBatchMetric()
        self.train_X_logp = SumExceptBatchMetric()
        self.train_E_logp = SumExceptBatchMetric()

    def __call__(self, masked_pred_X, masked_pred_E, true_X, true_E, log=True):
        """Compute training loss for protein design.
        
        Args:
            masked_pred_X: Predicted node features (amino acids)
            masked_pred_E: Predicted edge features (contacts)
            true_X: Ground truth node features
            true_E: Ground truth edge features
            log: Whether to log metrics
        """
        # Compute KL divergence for nodes (amino acids)
        X_kl = self.train_X_kl(masked_pred_X, true_X)
        E_kl = self.train_E_kl(masked_pred_E, true_E)
        
        # Compute log probability
        X_logp = self.train_X_logp(masked_pred_X, true_X)
        E_logp = self.train_E_logp(masked_pred_E, true_E)
        
        # Combine losses with weights
        loss = (
            self.lambda_train['X'] * X_kl +
            self.lambda_train['E'] * E_kl
        )
        
        if log:
            print(f"Train X KL: {X_kl:.2f} -- Train E KL: {E_kl:.2f}")
            print(f"Train X logp: {X_logp:.2f} -- Train E logp: {E_logp:.2f}")
        
        return loss

    def reset(self):
        self.train_X_kl.reset()
        self.train_E_kl.reset()
        self.train_X_logp.reset()
        self.train_E_logp.reset() 