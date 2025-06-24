import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import os
import gc
import torch.nn as nn
from torch_geometric.utils import to_dense_batch, to_dense_adj
import torch_geometric.utils

from .transformer import ProteinDenoiser
from .transition_model import ProteinTransitionModel
from metrics.protein_metrics_train import TrainLossDiscrete
from metrics.protein_metrics_sampling import SamplingProteinMetrics, SumExceptBatchKL, NLL
import utils as utils
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

        # Validation/Test metrics (matching Graph-DiT)
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = TrainLossDiscrete(lambda_train=cfg.model.lambda_train).train_X_logp
        self.val_E_logp = TrainLossDiscrete(lambda_train=cfg.model.lambda_train).train_E_logp
        self.val_y_collection = []

        # Test metrics (matching Graph-DiT)
        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = TrainLossDiscrete(lambda_train=cfg.model.lambda_train).train_X_logp
        self.test_E_logp = TrainLossDiscrete(lambda_train=cfg.model.lambda_train).train_E_logp
        self.test_y_collection = []
        self.sampling_metrics = SamplingProteinMetrics(
            dataset_infos,
            [],  # train_sequences, fill as needed
            []   # reference_sequences, fill as needed
        )

    def forward(self, noisy_data, unconditioned=False):
        # Extract data from noisy_data
        x = noisy_data['X_t'].float()  # [B, max_nodes, Xdim]
        e = noisy_data['E_t'].float()  # [total_edges, Edim] - sparse kNN edges
        node_mask = noisy_data['node_mask']  # [B, max_nodes]
        t = noisy_data['t']  # [B, 1] - time tensor
        edge_index = noisy_data.get('edge_index', None)  # [2, total_edges] - for sparse edges
        batch = noisy_data.get('batch', None)  # [total_nodes] - for sparse edges
        
        print(f"[DEBUG] Forward method - t shape: {t.shape}", flush=True)
        print(f"[DEBUG] Forward method - x shape: {x.shape}", flush=True)
        print(f"[DEBUG] Forward method - e shape: {e.shape}", flush=True)
        
        # Create sequence features (same as node features for now)
        p = x.clone()  # [B, max_nodes, Xdim] - sequence features same as node features
        
        # Graph-DiT approach: create dummy y if not present
        if 'y_t' in noisy_data:
            y = noisy_data['y_t'].float().clone()
        else:
            # Create dummy y tensor for unsupervised protein generation
            batch_size = x.size(0)
            y = torch.zeros(batch_size, self.ydim, device=x.device)
        
        # Pass arguments in correct order: (x, e, p, node_mask, y, t, unconditioned, edge_index, batch)
        pred = self.denoiser(x, e, p, node_mask, y, t, unconditioned=unconditioned, edge_index=edge_index, batch=batch)
        return pred

    def training_step(self, data, i):
        try:
            print("[DEBUG] Entered training_step", flush=True)
            print(f"[DEBUG] Data type: {type(data)}", flush=True)
            # Handle both single Data objects and batches
            if isinstance(data, list):
                print(f"[DEBUG] Got list with {len(data)} items", flush=True)
                protein_data = data[0]
                batch_size = 1
            else:
                batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
                print(f"[DEBUG] Processing batch with {batch_size} proteins", flush=True)
                protein_data = data
            # Convert to one-hot like Graph-DiT
            data_x = F.one_hot(protein_data.x.long().squeeze(-1), num_classes=self.Xdim).float()
            data_edge_attr = F.one_hot(protein_data.edge_attr.long(), num_classes=self.Edim).float()
            print(f"[DEBUG] Original data_x shape: {data_x.shape}", flush=True)
            print(f"[DEBUG] Original data_edge_attr shape: {data_edge_attr.shape}", flush=True)
            # For kNN, keep edges sparse! Don't convert to dense
            # Get actual max nodes from the batch for node features only
            if isinstance(data, list):
                max_nodes = max([d.num_nodes for d in data])
            else:
                # For batched PyG data, get max nodes from the batch
                batch_tensor = protein_data.batch
                max_nodes = 0
                for i in range(batch_size):
                    protein_nodes = (batch_tensor == i).sum().item()
                    max_nodes = max(max_nodes, protein_nodes)
            print(f"[DEBUG] Actual max_nodes in batch: {max_nodes}", flush=True)
            # Convert only node features to dense, keep edges sparse
            X, node_mask = to_dense_batch(data_x, protein_data.batch, max_num_nodes=max_nodes)
            print(f"[DEBUG] Dense X shape: {X.shape}", flush=True)
            print(f"[DEBUG] Sparse E shape: {data_edge_attr.shape}", flush=True)
            print(f"[DEBUG] Node mask shape: {node_mask.shape}", flush=True)
            # Store sparse edge data for apply_noise
            self.data = protein_data
            # Apply noise using sparse kNN approach
            noisy_data = self.apply_noise(X, data_edge_attr, node_mask, edge_index=protein_data.edge_index, batch=protein_data.batch)
            print("[DEBUG] Returned from apply_noise", flush=True)
            pred = self.forward(noisy_data)
            print("[DEBUG] Returned from forward", flush=True)
            # Now compute loss with sparse edges
            loss = self.loss_fn(
                masked_pred_X=pred.X,
                masked_pred_E=pred.E,
                true_X=X,
                true_E=data_edge_attr,
                edge_index=noisy_data['edge_index'],
                batch=noisy_data['batch']
            )
            print(f"[DEBUG] Loss computed: {loss}", flush=True)
            # Log loss with actual batch size
            self.log('loss', loss, batch_size=batch_size, sync_dist=True)
            return {'loss': loss}
        except Exception as e:
            print(f"[ERROR] Exception in training_step: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

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

    def apply_noise(self, X, E, node_mask, edge_index=None, batch=None):
        """Apply noise to node and edge features using sparse kNN approach."""
        try:
            print(f"[DEBUG] apply_noise called with X shape: {X.shape}, E shape: {E.shape}", flush=True)
            print(f"[DEBUG] node_mask shape: {node_mask.shape}", flush=True)
            print(f"[DEBUG] edge_index provided: {edge_index is not None}", flush=True)
            print(f"[DEBUG] batch provided: {batch is not None}", flush=True)
            
            batch_size = X.size(0)
            device = X.device
            
            # Get time step for noise application
            t = torch.randint(0, self.T, (batch_size,), device=device)
            t_normalized = t.float() / self.T
            alpha_bar_t = self.noise_schedule.get_alpha_bar(t_normalized)
            
            print(f"[DEBUG] Using sparse kNN approach", flush=True)
            print(f"[DEBUG] alpha_bar_t: {alpha_bar_t}", flush=True)
            # For sparse kNN edges, process each protein separately
            X_t_list = []
            E_t_list = []
            
            # Use provided edge_index and batch, or fall back to self.data if available
            if edge_index is None and self.data is not None:
                edge_index = self.data.edge_index  # [2, total_edges]
                batch_tensor = self.data.batch  # [total_nodes]
            elif edge_index is not None and batch is not None:
                edge_index = edge_index
                batch_tensor = batch
            else:
                # If no edge information is available, we need to create it
                # This is a fallback for cases where edge information is missing
                print(f"[WARNING] No edge information available, creating dummy edges", flush=True)
                # Create a simple edge index for the current batch
                total_nodes = X.size(0) * X.size(1)
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                batch_tensor = torch.arange(X.size(0), device=device).repeat_interleave(X.size(1))
            
            print(f"[DEBUG] edge_index shape: {edge_index.shape}", flush=True)
            print(f"[DEBUG] batch_tensor shape: {batch_tensor.shape}", flush=True)
            
            for i in range(batch_size):
                # Get current protein
                X_protein = X[i:i+1]  # [1, max_nodes, Xdim]
                mask_protein = node_mask[i:i+1]  # [1, max_nodes]
                alpha_i = alpha_bar_t[i]
                print(f"\nProcessing protein {i}:", flush=True)
                print(f"X_protein shape: {X_protein.shape}", flush=True)
                print(f"alpha_i: {alpha_i}", flush=True)
                
                # Get edges for this protein by checking which edges have both source and target in this protein
                # First, get the node indices for this protein
                protein_node_mask = (batch_tensor == i)  # [total_nodes]
                protein_node_indices = torch.where(protein_node_mask)[0]  # Indices of nodes in this protein
                
                if len(protein_node_indices) > 0:
                    # Get the start and end indices for this protein's nodes
                    start_idx = protein_node_indices[0]
                    end_idx = protein_node_indices[-1] + 1
                    # Filter edges where both source and target are in this protein
                    src_in_protein = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
                    dst_in_protein = (edge_index[1] >= start_idx) & (edge_index[1] < end_idx)
                    protein_edge_mask = src_in_protein & dst_in_protein
                    edge_indices = edge_index[:, protein_edge_mask]  # [2, num_edges_in_protein]
                    edge_features = E[protein_edge_mask]  # [num_edges_in_protein, Edim]
                else:
                    # No nodes for this protein, create empty edges
                    edge_indices = torch.zeros((2, 0), dtype=torch.long, device=device)
                    edge_features = torch.zeros((0, E.size(-1)), device=device)
                
                print(f"Protein {i} edges: {edge_indices.shape}, features: {edge_features.shape}", flush=True)
                
                # Apply noise to node features
                uniform_X = torch.ones_like(X_protein) / self.Xdim
                X_t = alpha_i * X_protein + (1 - alpha_i) * uniform_X
                
                # Apply noise to edge features
                if edge_features.size(0) > 0:
                    uniform_E = torch.ones_like(edge_features) / self.Edim
                    E_t = alpha_i * edge_features + (1 - alpha_i) * uniform_E
                else:
                    E_t = edge_features  # Keep empty
                
                # Store results
                X_t_list.append(X_t)
                E_t_list.append(E_t)
            
            # Concatenate results
            X_t = torch.cat(X_t_list, dim=0)  # [batch_size, max_nodes, Xdim]
            E_t = torch.cat(E_t_list, dim=0)  # [total_edges, Edim] - keep sparse
            print(f"[DEBUG] Final X_t shape: {X_t.shape}", flush=True)
            print(f"[DEBUG] Final E_t shape: {E_t.shape}", flush=True)
            
            # Keep everything sparse! No dense conversion needed
            # The denoiser will handle sparse edges directly
            return {
                'X_t': X_t,  # [B, max_nodes, Xdim] - dense node features
                'E_t': E_t,  # [total_edges, Edim] - sparse kNN edges
                't': t,
                'node_mask': node_mask,
                'edge_index': edge_index,  # [2, total_edges] - for sparse processing
                'batch': batch_tensor  # [total_nodes] - for sparse processing
            }
        except Exception as e:
            print(f"[ERROR] Exception in apply_noise: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

    def sample_batch(self, batch_id, batch_size, y, keep_chain, number_chain_steps, save_final, num_nodes=None):
        """Sample a batch of proteins."""
        # Initialize with random noise
        X = torch.randn(batch_size, self.max_n_nodes, self.Xdim, device=self.device)
        E = torch.randn(batch_size, self.max_n_nodes, self.max_n_nodes, self.Edim, device=self.device)
        y = y.to(self.device)
        
        # Create node mask
        if num_nodes is None:
            num_nodes = torch.randint(1, self.max_n_nodes + 1, (batch_size,), device=self.device)
        node_mask = torch.zeros(batch_size, self.max_n_nodes, dtype=torch.bool, device=self.device)  # Standardize as bool
        for i in range(batch_size):
            node_mask[i, :num_nodes[i]] = True
        
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
            X = X * node_mask.unsqueeze(-1).float()  # Cast to float for multiplication
            E = E * node_mask.unsqueeze(-1).unsqueeze(-1).float()  # Cast to float for multiplication
        
        return X, E, y, node_mask 

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_y_collection = []

    def on_validation_epoch_end(self) -> None:
        metrics = [
            self.val_nll.compute(),
            self.val_X_kl.compute() * self.T,
            self.val_E_kl.compute() * self.T,
            self.val_X_logp.compute(),
            self.val_E_logp.compute()
        ]
        if self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]:
            print(f"Epoch {self.current_epoch}: Val NLL {metrics[0]:.2f} -- Val Atom type KL {metrics[1]:.2f} -- ",
                  f"Val Edge type KL: {metrics[2]:.2f}", 'Val loss: %.2f \t Best :  %.2f\n' % (metrics[0], self.best_val_nll))
            
            # Milestone sampling like Graph-DiT
            try:
                print(f"Starting milestone sampling at epoch {self.current_epoch}...")
                self._do_milestone_sampling()
                print(f"Milestone sampling completed at epoch {self.current_epoch}")
            except Exception as e:
                print(f"Warning: Milestone sampling failed at epoch {self.current_epoch}: {str(e)}")
                print("Continuing training without milestone sampling...")
        
        self.log("val/NLL", metrics[0], sync_dist=True)
        self.log("val/X_KL", metrics[1], sync_dist=True)
        self.log("val/E_KL", metrics[2], sync_dist=True)
        self.log("val/X_logp", metrics[3], sync_dist=True)
        self.log("val/E_logp", metrics[4], sync_dist=True)
        if metrics[0] < self.best_val_nll:
            self.best_val_nll = metrics[0]
        self.val_counter += 1 

    def _do_milestone_sampling(self):
        """Generate samples at training milestones (25%, 50%, 75%, 100%) like Graph-DiT."""
        try:
            # Generate a small number of samples for milestone evaluation
            samples, all_ys = [], []
            num_samples = min(8, self.cfg.general.samples_to_generate)  # Small number for safety
            
            print(f"Generating {num_samples} samples for milestone evaluation...")
            
            for i in range(num_samples):
                try:
                    # Generate dummy y (you can modify this based on your needs)
                    y = torch.zeros(1, self.ydim, device=self.device)
                    
                    # Generate sample with error handling
                    X_sample, E_sample, y_sample, node_mask = self.sample_batch(
                        batch_id=i, 
                        batch_size=1, 
                        y=y, 
                        keep_chain=1, 
                        number_chain_steps=self.number_chain_steps, 
                        save_final=1
                    )
                    
                    samples.append((X_sample, E_sample))
                    all_ys.append(y_sample)
                    
                except Exception as e:
                    print(f"Warning: Failed to generate sample {i}: {str(e)}")
                    continue
            
            if samples:
                print(f"Successfully generated {len(samples)} samples, computing metrics...")
                # Compute sampling metrics safely
                try:
                    self.sampling_metrics(samples, all_ys, self.name, self.current_epoch, test=False)
                    print(f"Milestone sampling metrics computed for epoch {self.current_epoch}")
                except Exception as e:
                    print(f"Warning: Sampling metrics computation failed: {str(e)}")
                    print("Continuing without sampling metrics...")
            else:
                print("Warning: No samples were generated successfully")
                
        except Exception as e:
            print(f"Error in milestone sampling: {str(e)}")
            # Don't raise the exception - just log and continue

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        # Prepare batch (handle both single Data and batch)
        if isinstance(data, list):
            protein_data = data[0]
            batch_size = 1
        else:
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
            protein_data = data
        # Convert to one-hot
        data_x = F.one_hot(protein_data.x.long().squeeze(-1), num_classes=self.Xdim).float()
        data_edge_attr = F.one_hot(protein_data.edge_attr.long(), num_classes=self.Edim).float()
        # Get max nodes for dense node features
        if isinstance(data, list):
            max_nodes = max([d.num_nodes for d in data])
        else:
            batch_tensor = protein_data.batch
            max_nodes = 0
            for i in range(batch_size):
                protein_nodes = (batch_tensor == i).sum().item()
                max_nodes = max(max_nodes, protein_nodes)
        X, node_mask = to_dense_batch(data_x, protein_data.batch, max_num_nodes=max_nodes)
        # Store sparse edge data for apply_noise
        self.data = protein_data
        # Apply noise
        noisy_data = self.apply_noise(X, data_edge_attr, node_mask)
        pred = self.forward(noisy_data)

        # Convert sparse edge ground truth to dense, matching training logic
        edge_index = protein_data.edge_index
        batch = protein_data.batch
        true_E = data_edge_attr
        pred_E = pred.E
        dense_E = torch.zeros(pred_E.size(0), pred_E.size(1), pred_E.size(2), pred_E.size(3), device=pred_E.device)
        for i in range(batch_size):
            protein_node_mask = (batch == i)
            protein_node_indices = torch.where(protein_node_mask)[0]
            protein_edge_mask = torch.isin(edge_index[0], protein_node_indices)
            protein_edges = edge_index[:, protein_edge_mask]
            protein_edge_features = true_E[protein_edge_mask]
            local_edge_index = torch.zeros_like(protein_edges)
            for j, global_idx in enumerate(protein_node_indices):
                local_edge_index[0][protein_edges[0] == global_idx] = j
                local_edge_index[1][protein_edges[1] == global_idx] = j
            dense_E[i, local_edge_index[0], local_edge_index[1]] = protein_edge_features

        # Debug prints for shapes
        print(f"[DEBUG] pred.X shape: {pred.X.shape}, X shape: {X.shape}")
        print(f"[DEBUG] pred.E shape: {pred_E.shape}, dense_E shape: {dense_E.shape}")

        # Use dense edge metric computation
        self.val_nll(pred.X, X)
        self.val_X_kl(pred.X, X)
        self.val_E_kl(pred_E, dense_E)
        self.val_X_logp(pred.X, X)
        self.val_E_logp(pred_E, dense_E)

        # Log NLL for this batch
        nll_value = self.val_nll.compute()
        self.log('val/NLL', nll_value, batch_size=batch_size, sync_dist=True)

        # Store y for sampling metrics if available
        if hasattr(protein_data, 'y'):
            self.val_y_collection.append(protein_data.y)
        return {'loss': nll_value}

    @torch.no_grad()
    def test_step(self, data, batch_idx):
        # Prepare batch (handle both single Data and batch)
        if isinstance(data, list):
            protein_data = data[0]
            batch_size = 1
        else:
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
            protein_data = data
        # Convert to one-hot
        data_x = F.one_hot(protein_data.x.long().squeeze(-1), num_classes=self.Xdim).float()
        data_edge_attr = F.one_hot(protein_data.edge_attr.long(), num_classes=self.Edim).float()
        # Get max nodes for dense node features
        if isinstance(data, list):
            max_nodes = max([d.num_nodes for d in data])
        else:
            batch_tensor = protein_data.batch
            max_nodes = 0
            for i in range(batch_size):
                protein_nodes = (batch_tensor == i).sum().item()
                max_nodes = max(max_nodes, protein_nodes)
        X, node_mask = to_dense_batch(data_x, protein_data.batch, max_num_nodes=max_nodes)
        # Store sparse edge data for apply_noise
        self.data = protein_data
        # Apply noise
        noisy_data = self.apply_noise(X, data_edge_attr, node_mask)
        pred = self.forward(noisy_data)

        # Convert sparse edge ground truth to dense, matching validation logic
        edge_index = protein_data.edge_index
        batch = protein_data.batch
        true_E = data_edge_attr
        pred_E = pred.E
        dense_E = torch.zeros(pred_E.size(0), pred_E.size(1), pred_E.size(2), pred_E.size(3), device=pred_E.device)
        for i in range(batch_size):
            protein_node_mask = (batch == i)
            protein_node_indices = torch.where(protein_node_mask)[0]
            protein_edge_mask = torch.isin(edge_index[0], protein_node_indices)
            protein_edges = edge_index[:, protein_edge_mask]
            protein_edge_features = true_E[protein_edge_mask]
            local_edge_index = torch.zeros_like(protein_edges)
            for j, global_idx in enumerate(protein_node_indices):
                local_edge_index[0][protein_edges[0] == global_idx] = j
                local_edge_index[1][protein_edges[1] == global_idx] = j
            dense_E[i, local_edge_index[0], local_edge_index[1]] = protein_edge_features

        # Debug prints for shapes
        print(f"[DEBUG] pred.X shape: {pred.X.shape}, X shape: {X.shape}")
        print(f"[DEBUG] pred.E shape: {pred_E.shape}, dense_E shape: {dense_E.shape}")

        # Use dense edge metric computation
        self.test_nll(pred.X, X)
        self.test_X_kl(pred.X, X)
        self.test_E_kl(pred_E, dense_E)
        self.test_X_logp(pred.X, X)
        self.test_E_logp(pred_E, dense_E)
        
        # Store y for sequence metrics if available
        if hasattr(protein_data, 'y'):
            self.test_y_collection.append(protein_data.y)
        nll_value = self.test_nll.compute()
        self.log('test_nll', nll_value, batch_size=batch_size, sync_dist=True)
        return {'loss': nll_value}

    def on_test_epoch_start(self) -> None:
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_collection = []

    def on_test_epoch_end(self):
        metrics = [
            self.test_nll.compute(),
            self.test_X_kl.compute() * self.T,
            self.test_E_kl.compute() * self.T,
            self.test_X_logp.compute(),
            self.test_E_logp.compute()
        ]
        print(f"Test NLL: {metrics[0]:.2f} -- Test Atom type KL: {metrics[1]:.2f} -- Test Edge type KL: {metrics[2]:.2f}")
        self.log("test/NLL", metrics[0], sync_dist=True)
        self.log("test/X_KL", metrics[1], sync_dist=True)
        self.log("test/E_KL", metrics[2], sync_dist=True)
        self.log("test/X_logp", metrics[3], sync_dist=True)
        self.log("test/E_logp", metrics[4], sync_dist=True)
        
        # Sequence-based metrics (validity, uniqueness, novelty) on generated samples
        # Here we assume you want to sample from the model for sequence metrics
        # You may want to adjust batch size and number of samples as needed
        samples, all_ys = [], []
        num_samples = 32  # or another number as appropriate
        for _ in range(num_samples):
            # Generate a batch of samples (adjust as needed)
            y = torch.zeros(1, self.ydim, device=self.device)  # dummy y
            X_sample, E_sample, y_sample, node_mask = self.sample_batch(
                batch_id=0, batch_size=1, y=y, keep_chain=1, number_chain_steps=self.number_chain_steps, save_final=1
            )
            samples.append((X_sample, E_sample))
            all_ys.append(y_sample)
        print("Computing sequence-based test metrics...")
        # Call the sampling metrics properly
        self.sampling_metrics(samples, all_ys, self.name, self.current_epoch, test=True) 