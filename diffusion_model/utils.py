import torch
from dataclasses import dataclass

@dataclass
class PlaceHolder:
    X: torch.Tensor
    E: torch.Tensor
    y: torch.Tensor = None

def sample_discrete_features(probX, probE, probP, node_mask):
    """Sample discrete features from probability distributions."""
    # Sample amino acids
    X = torch.multinomial(probX.reshape(-1, probX.size(-1)), 1).reshape(probX.size(0), -1)
    
    # Sample edges
    E = torch.multinomial(probE.reshape(-1, probE.size(-1)), 1).reshape(probE.size(0), -1)
    
    # Sample sequence
    P = torch.multinomial(probP.reshape(-1, probP.size(-1)), 1).reshape(probP.size(0), -1)
    
    # Apply node mask
    X = X * node_mask
    E = E * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    P = P * node_mask
    
    return PlaceHolder(X=X, E=E, y=None) 