import torch
from torch.nn import functional as F
import numpy as np
from protein_dit.utils import PlaceHolder


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=30, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_discrete_features(probX, probE, probP, node_mask, step=None, add_noise=True):
    '''Sample features from multinomial distribution with given probabilities
    Args:
        probX: bs, n, dx_out        node features (amino acids)
        probE: bs, n, n, de_out     edge features (distances)
        probP: bs, n, dp_out        sequence features
        node_mask: bs, n            node mask
    '''
    bs, n, _ = probX.shape

    # Sample amino acids
    probX[~node_mask] = 1 / probX.shape[-1]
    probX = probX.reshape(bs * n, -1)
    probX = probX + 1e-12
    probX = probX / probX.sum(dim=-1, keepdim=True)
    X_t = probX.multinomial(1).reshape(bs, n)

    # Sample edges
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)
    probE = probE + 1e-12
    probE = probE / probE.sum(dim=-1, keepdim=True)

    # Sample edges
    E_t = probE.multinomial(1).reshape(bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    # Sample sequence
    probP[~node_mask] = 1 / probP.shape[-1]
    probP = probP.reshape(bs * n, -1)
    probP = probP + 1e-12
    probP = probP / probP.sum(dim=-1, keepdim=True)
    P_t = probP.multinomial(1).reshape(bs, n)

    return PlaceHolder(X=X_t, E=E_t, P=P_t, y=torch.zeros(bs, 0).type_as(X_t))


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ Compute posterior distribution for X, E, and P
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    X_t = X_t.float()
    Qt_T = Qt.transpose(-1, -2).float()
    assert Qt.dim() == 3
    left_term = X_t @ Qt_T
    left_term = left_term.unsqueeze(dim=2)
    right_term = Qsb.unsqueeze(1)
    numerator = left_term * right_term
    
    denominator = Qtb @ X_t.transpose(-1, -2)
    denominator = denominator.transpose(-1, -2)
    denominator = denominator.unsqueeze(-1)

    denominator[denominator == 0] = 1.
    return numerator / denominator


def mask_distributions(true_X, true_E, true_P, pred_X, pred_E, pred_P, node_mask):
    # Add a small value everywhere to avoid nans
    pred_X = pred_X.clamp_min(1e-18)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)

    pred_E = pred_E.clamp_min(1e-18)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    pred_P = pred_P.clamp_min(1e-18)
    pred_P = pred_P / torch.sum(pred_P, dim=-1, keepdim=True)

    # Set masked rows to arbitrary distributions
    row_X = torch.ones(true_X.size(-1), dtype=true_X.dtype, device=true_X.device)
    row_E = torch.zeros(true_E.size(-1), dtype=true_E.dtype, device=true_E.device).clamp_min(1e-18)
    row_E[0] = 1.
    row_P = torch.ones(true_P.size(-1), dtype=true_P.dtype, device=true_P.device)

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    true_P[~node_mask] = row_P
    pred_X[~node_mask] = row_X
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_P[~node_mask] = row_P

    return true_X, true_E, true_P, pred_X, pred_E, pred_P


def posterior_distributions(X, X_t, Qt, Qsb, Qtb, X_dim, E_dim, P_dim):
    """Compute posterior distributions for X (amino acids), E (edges), and P (sequence)"""
    bs, n, d = X.shape
    X = X.float()
    Qt_X_T = torch.transpose(Qt.X, -2, -1).float()
    left_term = X_t @ Qt_X_T
    right_term = X @ Qsb.X
    
    numerator = left_term * right_term
    denominator = X @ Qtb.X
    denominator = denominator * X_t
    
    # Split dimensions for X, E, and P
    num_X = numerator[:, :, :X_dim]
    num_E = numerator[:, :, X_dim:X_dim+E_dim].reshape(bs, n*n, -1)
    num_P = numerator[:, :, X_dim+E_dim:].reshape(bs, n, -1)

    deno_X = denominator[:, :, :X_dim]
    deno_E = denominator[:, :, X_dim:X_dim+E_dim].reshape(bs, n*n, -1)
    deno_P = denominator[:, :, X_dim+E_dim:].reshape(bs, n, -1)

    deno_X = deno_X.sum(dim=-1).unsqueeze(-1)
    deno_E = deno_E.sum(dim=-1).unsqueeze(-1)
    deno_P = deno_P.sum(dim=-1).unsqueeze(-1)

    deno_X[deno_X == 0.] = 1
    deno_E[deno_E == 0.] = 1
    deno_P[deno_P == 0.] = 1

    prob_X = num_X / deno_X
    prob_E = num_E / deno_E
    prob_P = num_P / deno_P
    
    prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True)
    prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True)
    prob_P = prob_P / prob_P.sum(dim=-1, keepdim=True)

    return PlaceHolder(X=prob_X, E=prob_E, P=prob_P, y=None)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    
    # Sample amino acid noise
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_X = F.one_hot(U_X.long(), num_classes=x_limit.shape[-1]).float()

    # Sample edge noise
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_E = F.one_hot(U_E.long(), num_classes=e_limit.shape[-1]).float()

    # Sample protein sequence noise
    p_limit = limit_dist.P[None, None, :].expand(bs, n_max, -1)
    U_P = p_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_P = F.one_hot(U_P.long(), num_classes=p_limit.shape[-1]).float()

    U_X = U_X.to(node_mask.device)
    U_E = U_E.to(node_mask.device)
    U_P = U_P.to(node_mask.device)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = (U_E + torch.transpose(U_E, 1, 2))

    assert (U_E == torch.transpose(U_E, 1, 2)).all()
    return PlaceHolder(X=U_X, E=U_E, P=U_P, y=None).mask(node_mask) 