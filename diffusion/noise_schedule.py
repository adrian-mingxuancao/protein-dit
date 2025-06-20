import torch
from utils import PlaceHolder
from diffusion import diffusion_utils
    
class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=1)
        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        self.alphas_bar = self.alphas_bar.to(t_int.device)
        return self.alphas_bar[t_int.long()]


class MarginalTransition:
    def __init__(self, x_marginals, e_marginals, xe_conditions, ex_conditions, p_marginals, y_classes, n_nodes):
        """
        Initialize transition model that captures all state dependencies:
        - Node → Node transitions (amino acid types)
        - Edge → Node transitions (structure to sequence)
        - Edge → Edge transitions (structural dependencies)
        - Node → Edge transitions (sequence to structure)
        - Protein sequence transitions (sequence dependencies)
        
        Args:
            x_marginals: Marginal distribution of amino acid types
            e_marginals: Marginal distribution of edge types (distance bins)
            xe_conditions: Conditional distribution of edge types given amino acid types
            ex_conditions: Conditional distribution of amino acid types given edge types
            p_marginals: Marginal distribution of protein sequence positions
            y_classes: Number of global feature classes
            n_nodes: Number of nodes in the graph
        """
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.P_classes = len(p_marginals)
        self.y_classes = y_classes
        self.n_nodes = n_nodes
        
        # Store marginal and conditional distributions
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals
        self.p_marginals = p_marginals
        self.xe_conditions = xe_conditions
        self.ex_conditions = ex_conditions

        # Create structure transitions (like Graph-DiT)
        self.u_struct = self.get_structure_transition(x_marginals, e_marginals, xe_conditions, ex_conditions, n_nodes)
        # Create sequence transitions
        self.u_seq = self.get_sequence_transition(p_marginals, n_nodes)

    def get_structure_transition(self, x_marginals, e_marginals, xe_conditions, ex_conditions, n_nodes):
        """Creates transition matrix for structure (atoms and edges)"""
        # Create transition matrices
        u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)  # 1, Dx, Dx
        u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)  # 1, De, De
        u_xe = xe_conditions.unsqueeze(0)  # 1, Dx, De
        u_ex = ex_conditions.unsqueeze(0)  # 1, De, Dx
        
        # Expand edge transitions for all node pairs
        u_e = u_e.repeat(1, n_nodes, n_nodes)  # (1, n*de, n*de)
        u_xe = u_xe.repeat(1, 1, n_nodes)  # (1, dx, n*de)
        u_ex = u_ex.repeat(1, n_nodes, 1)  # (1, n*de, dx)
        
        # Combine atom-edge transitions
        u0 = torch.cat([u_x, u_xe], dim=2)  # (1, dx, dx + n*de)
        u1 = torch.cat([u_ex, u_e], dim=2)  # (1, n*de, dx + n*de)
        return torch.cat([u0, u1], dim=1)  # (1, dx + n*de, dx + n*de)

    def get_sequence_transition(self, p_marginals, n_nodes):
        """Creates transition matrix for protein sequence"""
        # Create protein sequence transitions
        u_p = p_marginals.unsqueeze(0).expand(self.P_classes, -1).unsqueeze(0)  # 1, Dp, Dp
        # Repeat for each node position
        u_p = u_p.repeat(1, n_nodes, 1)  # (1, n*dp, dp)
        # Create identity matrix for each position
        eye = torch.eye(self.P_classes, device=u_p.device).unsqueeze(0)  # (1, Dp, Dp)
        eye = eye.repeat(1, n_nodes, 1)  # (1, n*dp, dp)
        return torch.cat([eye, u_p], dim=2)  # (1, n*dp, 2*dp)

    def index_edge_margin(self, X, q_e, n_bond=5):
        """Index edge marginals based on node types"""
        # Get node types
        node_types = X.argmax(dim=-1)  # [B, N]
        # Get edge marginals for each node pair
        edge_marginals = self.e_marginals[node_types]  # [B, N, E]
        edge_marginals = edge_marginals.unsqueeze(1)  # [B, 1, N, E]
        edge_marginals = edge_marginals.expand(-1, X.size(1), -1, -1)  # [B, N, N, E]
        # Combine with edge transitions
        q_e = q_e * edge_marginals
        return q_e

    def get_Qt(self, beta_t, device):
        """Returns one-step transition matrices for X, E, and P"""
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        
        # Structure transitions
        u_struct = self.u_struct.to(device)
        q_struct = beta_t * u_struct + (1 - beta_t) * torch.eye(u_struct.size(-1), device=device).unsqueeze(0)
        
        # Sequence transitions
        u_seq = self.u_seq.to(device)
        q_seq = beta_t * u_seq + (1 - beta_t) * torch.eye(u_seq.size(-1), device=device).unsqueeze(0)
        
        return PlaceHolder(X=q_struct, E=None, P=q_seq, y=None)

    def get_Qt_bar(self, alpha_bar_t, device):
        """Returns t-step transition matrices for X, E, and P"""
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        
        # Structure transitions
        u_struct = self.u_struct.to(device)
        q_struct = alpha_bar_t * torch.eye(u_struct.size(-1), device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_struct
        
        # Sequence transitions
        u_seq = self.u_seq.to(device)
        q_seq = alpha_bar_t * torch.eye(u_seq.size(-1), device=device).unsqueeze(0) + (1 - alpha_bar_t) * u_seq
        
        return PlaceHolder(X=q_struct, E=None, P=q_seq, y=None) 