# These imports are tricky because they use c++, do not move them
import os, shutil
import warnings

import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
import pytorch_lightning as pl

import utils
from datasets import ProteinDataModule, DatasetInfo
from diffusion_model import Protein_Graph_DiT
from metrics.protein_metrics_train import TrainLossDiscrete
from metrics.protein_metrics_sampling import SamplingProteinMetrics
from analysis.visualization import ProteinVisualization

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")

@hydra.main(
    version_base="1.1", config_path="configs", config_name="train_config"
)
def main(cfg: DictConfig):
    # Initialize protein-specific data module
    datamodule = ProteinDataModule(cfg)
    datamodule.prepare_data()
    dataset_infos = DatasetInfo(datamodule.train_dataset)
    
    # Initialize protein-specific metrics
    train_metrics = TrainLossDiscrete(cfg.model.lambda_train)
    sampling_metrics = SamplingProteinMetrics(
        dataset_infos,
        datamodule.train_dataset,
        datamodule.val_dataset
    )

    # Set working directory for resume if needed
    if cfg.general.test_only:
        # Test-only mode
        print(f"Running test-only mode with checkpoint: {cfg.general.test_only}")
        os.chdir(os.path.dirname(cfg.general.test_only))
        model = Protein_Graph_DiT.load_from_checkpoint(
            cfg.general.test_only,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            cfg=cfg
        )
    elif cfg.general.resume is not None and os.path.exists(cfg.general.resume):
        print(f"Resuming from checkpoint: {cfg.general.resume}")
        os.chdir(os.path.dirname(cfg.general.resume))
        model = Protein_Graph_DiT.load_from_checkpoint(
            cfg.general.resume,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            cfg=cfg
        )
    else:
        model = Protein_Graph_DiT(
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            cfg=cfg
        )
    
    # Configure trainer
    trainer_kwargs = dict(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val/NLL",
                patience=cfg.train.early_stopping_patience,
                mode="min",
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=cfg.logging.save_dir + "/checkpoints",
                filename="{epoch:02d}-{val_NLL:.2f}",
                save_top_k=cfg.logging.save_top_k,
                monitor="val/NLL",
                mode=cfg.logging.mode,
            ),
        ],
        limit_val_batches=cfg.train.limit_val_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        val_check_interval=cfg.logging.val_check_interval,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )
    if hasattr(cfg.train, 'strategy'):
        trainer_kwargs['strategy'] = cfg.train.strategy
    trainer = Trainer(**trainer_kwargs)

    # Train and test
    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.save_model:
            trainer.save_checkpoint(f"checkpoints/{cfg.general.name}/last.ckpt")
        print("Training completed successfully!")
        print("To run testing separately, use: python main.py ++general.test_only=<checkpoint_path>")
        # Note: Testing is now separate from training to avoid interrupting training progress
        # Uncomment the line below if you want to run testing immediately after training
        # trainer.test(model, datamodule=datamodule)
    else:
        # Test-only mode
        print(f"Running test-only mode with checkpoint: {cfg.general.test_only}")
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)

if __name__ == "__main__":
    main() 