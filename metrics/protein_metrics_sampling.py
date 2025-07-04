import torch
import numpy as np

class SumExceptBatchMetric:
    def __init__(self):
        self.reset()
    
    def __call__(self, pred, true):
        """Compute metric summed over all dimensions except batch and accumulate."""
        metric = torch.sum(pred * true, dim=-1)
        
        # Accumulate for averaging
        self.total += metric.sum().item()
        self.count += 1
        
        return metric.sum()  # Return scalar for immediate use
    
    def reset(self):
        self.total = 0
        self.count = 0
    
    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

class SumExceptBatchKL:
    def __init__(self):
        self.reset()
    
    def __call__(self, pred, true):
        """Compute KL divergence using PyTorch's F.kl_div (like Graph-DiT)."""
        # Debug prints to understand tensor shapes
        print(f"[DEBUG KL] pred shape: {pred.shape}, true shape: {true.shape}")
        print(f"[DEBUG KL] pred last dim: {pred.size(-1)}, true last dim: {true.size(-1)}")
        
        # Apply softmax to predictions to get probabilities
        pred_probs = torch.softmax(pred, dim=-1)
        
        # Use PyTorch's F.kl_div for proper KL divergence computation
        # F.kl_div expects log-probabilities for predictions and probabilities for targets
        pred_log_probs = torch.log_softmax(pred, dim=-1)
        
        # Compute KL divergence using PyTorch's implementation
        kl = torch.nn.functional.kl_div(pred_log_probs, true, reduction='none')
        kl = torch.sum(kl, dim=-1)  # Sum over feature dimensions
        kl = torch.mean(kl)  # Average over batch
        
        # Accumulate for averaging across validation steps
        self.total += kl.item()
        self.count += 1
        
        return kl  # Return scalar for immediate use
    
    def reset(self):
        self.total = 0
        self.count = 0
    
    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

class NLL:
    def __init__(self):
        self.reset()
    
    def __call__(self, pred, true):
        """Compute negative log likelihood using PyTorch's cross_entropy (like Graph-DiT)."""
        # Use PyTorch's cross_entropy for proper NLL computation
        # Convert one-hot targets to class indices
        target_indices = torch.argmax(true, dim=-1)
        
        # Compute cross-entropy loss (which is NLL for one-hot targets)
        nll = torch.nn.functional.cross_entropy(pred, target_indices, reduction='none')
        nll = torch.mean(nll)  # Average over batch
        
        # Accumulate for averaging across validation steps
        self.total += nll.item()
        self.count += 1
        
        return nll  # Return scalar for immediate use
    
    def reset(self):
        self.total = 0
        self.count = 0
    
    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

class SamplingProteinMetrics:
    def __init__(self, dataset_infos, train_sequences, reference_sequences):
        self.dataset_infos = dataset_infos
        self.train_sequences = train_sequences
        self.reference_sequences = reference_sequences
        self.sampling_X_kl = SumExceptBatchMetric()
        self.sampling_E_kl = SumExceptBatchMetric()
        self.sampling_X_logp = SumExceptBatchMetric()
        self.sampling_E_logp = SumExceptBatchMetric()

    def __call__(self, samples, all_ys, name, current_epoch, val_counter=-1, test=False):
        """Compute sampling metrics for protein design.
        
        Args:
            samples: Generated protein samples
            all_ys: Target properties
            name: Model name
            current_epoch: Current training epoch
            val_counter: Validation counter
            test: Whether this is a test run
        """
        # Convert samples to sequences
        sequences = self.convert_samples_to_sequences(samples)
        
        # Compute metrics
        X_kl = self.sampling_X_kl.compute()
        E_kl = self.sampling_E_kl.compute()
        X_logp = self.sampling_X_logp.compute()
        E_logp = self.sampling_E_logp.compute()
        
        # Compute sequence-based metrics
        validity = self.compute_validity(sequences)
        uniqueness = self.compute_uniqueness(sequences)
        novelty = self.compute_novelty(sequences)
        
        # Log metrics
        print(f"\nSampling metrics for epoch {current_epoch}:")
        print(f"X KL: {X_kl:.2f} -- E KL: {E_kl:.2f}")
        print(f"X logp: {X_logp:.2f} -- E logp: {E_logp:.2f}")
        print(f"Validity: {validity:.2f}")
        print(f"Uniqueness: {uniqueness:.2f}")
        print(f"Novelty: {novelty:.2f}")
        
        return sequences

    def reset(self):
        self.sampling_X_kl.reset()
        self.sampling_E_kl.reset()
        self.sampling_X_logp.reset()
        self.sampling_E_logp.reset()

    def convert_samples_to_sequences(self, samples):
        """Convert generated samples to amino acid sequences."""
        sequences = []
        for sample in samples:
            # Convert node features to amino acid sequence
            sequence = self.dataset_infos.convert_node_features_to_sequence(sample[0])
            sequences.append(sequence)
        return sequences

    def compute_validity(self, sequences):
        """Compute the fraction of valid protein sequences."""
        valid_count = sum(1 for seq in sequences if self.is_valid_sequence(seq))
        return valid_count / len(sequences) if sequences else 0.0

    def compute_uniqueness(self, sequences):
        """Compute the fraction of unique sequences."""
        unique_sequences = set(sequences)
        return len(unique_sequences) / len(sequences) if sequences else 0.0

    def compute_novelty(self, sequences):
        """Compute the fraction of sequences not in the training set."""
        novel_count = sum(1 for seq in sequences if seq not in self.train_sequences)
        return novel_count / len(sequences) if sequences else 0.0

    def is_valid_sequence(self, sequence):
        """Check if a sequence is valid."""
        # Add your sequence validation logic here
        # For example, check if it contains only valid amino acids
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        return all(aa in valid_aas for aa in sequence) 