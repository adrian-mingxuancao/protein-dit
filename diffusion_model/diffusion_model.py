import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import os
import gc
import torch.nn as nn

from .transformer import ProteinDenoiser
from .transition_model import ProteinTransitionModel
from metrics.protein_metrics_train import TrainLossDiscrete
from metrics.protein_metrics_sampling import SamplingProteinMetrics
import protein_dit.utils as utils
from .noise_schedule import PredefinedNoiseScheduleDiscrete

class Protein_Graph_DiT(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics):
        super().__init__()
        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.test_only = cfg.general.test_only

        # Store dataset info
        self.dataset_info = dataset_infos
        self.max_n_nodes = dataset_infos.max_n_nodes
        self.Xdim = 20  # Number of amino acids
        self.Edim = cfg.model.transition_model.e_classes  # Use e_classes from transition model
        self.ydim = cfg.model.transition_model.y_classes  # Use y_classes from transition model
        self.Xdim_output = dataset_infos.output_dims['X']
        self.Edim_output = dataset_infos.output_dims['E']
        self.ydim_output = dataset_infos.output_dims['y']

        # Store config
        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps
        self.guide_scale = cfg.model.guide_scale

        # Initialize noise schedule
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps
        )

        # Initialize transition model
        self.transition_model = ProteinTransitionModel(
            x_marginals=dataset_infos.node_types.float() / torch.sum(dataset_infos.node_types.float()),
            e_marginals=dataset_infos.edge_types.float() / torch.sum(dataset_infos.edge_types.float()),
            p_marginals=dataset_infos.position_types.float() / torch.sum(dataset_infos.position_types.float()),
            xe_conditions=dataset_infos.transition_xe,
            ex_conditions=dataset_infos.transition_ex,
            y_classes=self.ydim,
            n_nodes=cfg.model.max_n_nodes
        )
        self.denoiser = ProteinDenoiser(
            max_n_nodes=cfg.model.max_n_nodes,
            hidden_size=cfg.model.hidden_size,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            drop_condition=cfg.model.drop_condition,
            Xdim=self.Xdim,
            Edim=self.Edim,
            Pdim=self.Xdim,  # Using same dimension as amino acids for sequence
            ydim=self.ydim
        )
        self.loss_fn = TrainLossDiscrete(lambda_train=cfg.model.lambda_train)
        self.data = None  # Will be set during training
        self.debug = True  # Enable debug prints

        # Training state
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.batch_size = cfg.train.batch_size

    def forward(self, noisy_data, unconditioned=False):
        x, e, y = noisy_data['X_t'].float(), noisy_data['E_t'].float(), noisy_data['y_t'].float().clone()
        node_mask, t = noisy_data['node_mask'], noisy_data['t']
        pred = self.denoiser(x, e, node_mask, y=y, t=t, unconditioned=unconditioned)
        return pred

    def training_step(self, data, i):
        # Convert to one-hot
        data_x = F.one_hot(data.x.long().squeeze(-1), num_classes=self.Xdim).float()
        data_edge_attr = F.one_hot(data.edge_attr.long(), num_classes=self.Edim).float()

        # Store data for apply_noise
        self.data = data

        # Apply noise
        noisy_data = self.apply_noise(data_x, data_edge_attr, data.y, data.batch)
        pred = self.forward(noisy_data)

        # Compute loss
        loss = self.loss_fn(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=data_x,
            true_E=data_edge_attr
        )

        self.log('loss', loss, batch_size=data_x.size(0), sync_dist=True)
        return {'loss': loss}

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay
        )
        return optimizer

    def on_fit_start(self) -> None:
        self.train_iterations = self.trainer.datamodule.training_iterations
        print('on fit train iteration:', self.train_iterations)
        print("Size of the input features Xdim {}, Edim {}, ydim {}".format(self.Xdim, self.Edim, self.ydim))

    def on_train_epoch_start(self) -> None:
        if self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]:
            print("Starting train epoch {}/{}...".format(self.current_epoch, self.trainer.max_epochs))
        self.start_epoch_time = time.time()
        self.loss_fn.reset()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]:
            log = True
        else:
            log = False
        self.loss_fn.log_epoch_metrics(self.current_epoch, self.start_epoch_time, log)

    def apply_noise(self, X, E, P, batch):
        """Apply noise to the data."""
        # Debug prints for input shapes
        print(f"\nInput shapes:")
        print(f"X shape: {X.shape}")
        print(f"E shape: {E.shape}")
        print(f"X total elements: {X.numel()}")
        print(f"E total elements: {E.numel()}")
        
        # Get timestep
        t = torch.randint(0, self.T, (X.size(0),), device=X.device)
        t_float = t / self.T
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        
        # Get edge index from data
        edge_index = None
        edge_method = 'radius'  # Default to radius
        if hasattr(self, 'data') and hasattr(self.data, 'edge_index'):
            edge_index = self.data.edge_index
            edge_method = 'knn'  # Use kNN when we have edge_index
            print(f"Got edge_index from data: shape={edge_index.shape}")  # Always print this
        else:
            print("No edge_index found in data")  # Always print this
        
        # Get transition probabilities
        transitions = self.transition_model.get_Qt_bar(
            alpha_bar_t, 
            X.device, 
            X.size(0),
            edge_index=edge_index,
            edge_method=edge_method  # Pass edge_method here
        )
        
        # Process each chunk
        X_t_list = []
        E_t_list = []
        
        for i, (Q_struct, Q_seq) in enumerate(transitions):
            # Get current chunk
            chunk_size = Q_struct.size(0)
            X_chunk = X[i:i+chunk_size]  # [B, n, X]
            E_chunk = E[i:i+chunk_size]  # [B, e, E] where e is number of edges
            
            print(f"\nProcessing chunk {i}:")
            print(f"X_chunk shape: {X_chunk.shape}")
            print(f"E_chunk shape: {E_chunk.shape}")
            print(f"Q_struct shape: {Q_struct.shape}")
            
            if edge_method == 'knn':
                # For kNN, Q_struct is sparse
                # Process amino acids
                n_nodes = X_chunk.size(1)  # Get number of nodes
                X_flat = X_chunk.reshape(-1, self.Xdim)  # [B*n, X]
                print(f"X_flat shape: {X_flat.shape}")
                
                # Process edges
                E_flat = E_chunk.reshape(-1, self.Edim)  # [B*e, E]
                print(f"E_flat shape: {E_flat.shape}")
                
                # Process amino acids
                X_probs = torch.sparse.mm(Q_struct, X_flat)  # [n, X]
                X_probs = X_probs.reshape(chunk_size, n_nodes, self.Xdim)  # [B, n, X]
                
                # Process edges
                E_probs = torch.sparse.mm(Q_struct, E_flat)  # [e, E]
                E_probs = E_probs.reshape(chunk_size, -1, self.Edim)  # [B, e, E]
                
                # Add cross-transitions
                X_to_E = torch.sparse.mm(Q_struct, X_flat)  # [n, X]
                X_to_E = X_to_E.reshape(chunk_size, n_nodes, self.Edim)  # [B, n, E]
                
                E_to_X = torch.sparse.mm(Q_struct, E_flat)  # [e, E]
                E_to_X = E_to_X.reshape(chunk_size, -1, self.Xdim)  # [B, e, X]
                
                # Combine structure probabilities
                X_t = X_probs + E_to_X
                E_t = E_probs + X_to_E  # [B, e, E]
            else:
                print(f"Processing chunk {i} with dense matrices")
                # For radius method, use dense matrices
                X_t = torch.matmul(Q_struct, X_chunk.reshape(chunk_size, -1, self.Xdim)).reshape(chunk_size, X_chunk.size(1), self.Xdim)
                E_t = torch.matmul(Q_struct, E_chunk.reshape(chunk_size, -1, self.Edim)).reshape(chunk_size, E_chunk.size(1), self.Edim)
            
            # Store results
            X_t_list.append(X_t)
            E_t_list.append(E_t)
        
        # Concatenate results
        X_t = torch.cat(X_t_list, dim=0)
        E_t = torch.cat(E_t_list, dim=0)
        
        return {
            'X_t': X_t,
            'E_t': E_t,
            't': t,
            'node_mask': self.data.node_mask if hasattr(self, 'data') else None
        }

    def sample_batch(self, batch_id, batch_size, y, keep_chain, number_chain_steps, save_final, num_nodes=None):
        """Sample a batch of proteins."""
        # Initialize with random noise
        X = torch.randn(batch_size, self.max_n_nodes, self.Xdim, device=self.device)
        E = torch.randn(batch_size, self.max_n_nodes, self.max_n_nodes, self.Edim, device=self.device)
        y = y.to(self.device)
        
        # Create node mask
        if num_nodes is None:
            num_nodes = torch.randint(1, self.max_n_nodes + 1, (batch_size,), device=self.device)
        node_mask = torch.zeros(batch_size, self.max_n_nodes, device=self.device)
        for i in range(batch_size):
            node_mask[i, :num_nodes[i]] = 1
        
        # Sampling loop
        for t in range(self.T - 1, -1, -1):
            # Get noisy data
            noisy_data = {
                'X_t': X,
                'E_t': E,
                'y_t': y,
                't': torch.full((batch_size,), t, device=self.device),
                'node_mask': node_mask
            }
            
            # Get predictions
            pred = self.forward(noisy_data)
            
            # Update samples
            X = pred.X
            E = pred.E
            
            # Apply node mask
            X = X * node_mask.unsqueeze(-1)
            E = E * node_mask.unsqueeze(-1).unsqueeze(-1)
        
        return X, E, y, node_mask 