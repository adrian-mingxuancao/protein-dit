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
    if cfg.logging.resume_training and os.path.exists(cfg.logging.resume_ckpt_path):
        os.chdir(os.path.dirname(cfg.logging.resume_ckpt_path))
    
    # Initialize model
    if cfg.logging.resume_training and os.path.exists(cfg.logging.resume_ckpt_path):
        print(f"Resuming from checkpoint: {cfg.logging.resume_ckpt_path}")
        model = Protein_Graph_DiT.load_from_checkpoint(
            cfg.logging.resume_ckpt_path,
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
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.train.early_stopping_patience,
                mode="min",
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=cfg.logging.save_dir + "/checkpoints",
                filename="{epoch:02d}-{val_loss:.2f}",
                save_top_k=cfg.logging.save_top_k,
                monitor=cfg.logging.monitor,
                mode=cfg.logging.mode,
            ),
        ],
        limit_val_batches=cfg.train.limit_val_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        val_check_interval=cfg.logging.val_check_interval,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )

    # Train and test
    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule)
        if cfg.general.save_model:
            trainer.save_checkpoint(f"checkpoints/{cfg.general.name}/last.ckpt")
        trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main() 