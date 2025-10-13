import hydra
from omegaconf import DictConfig, OmegaConf
from primal.data import mixed_motion_dm
from primal.utils.pylogger import get_pylogger
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import os, sys, glob
import shutil

            

log = get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(version_base=None, config_path="../primal/configs", 
            config_name="train_diffusion")
def train(cfg: DictConfig) -> None:
    """Main training function for motion diffusion models.
    
    Handles three training modes:
    1. Fresh training from scratch
    2. Resume training from a previous checkpoint 
    3. Fine-tune from a pretrained base model
    
    Args:
        cfg (DictConfig): Hydra configuration object containing:
            - data: Dataset configuration (AMASS, batch size, etc.)
            - model: Model architecture and hyperparameters
            - trainer: PyTorch Lightning trainer settings
            - logger: Logging configuration (tensorboard, etc.)
            - resume_from_exp (optional): Path to experiment for resuming training
            - finetune_from_exp (optional): Path to pretrained model for fine-tuning
    """
    # when resume training, everything (model params and hparams, except devices) will be loaded from the checkpoint
    if hasattr(cfg, 'resume_from_exp'):
        # update the setting
        cfg_path = os.path.join(cfg.resume_from_exp, '.hydra', 'config.yaml')
        deviceconfig = cfg.trainer.devices
        resume_from_exp = cfg.resume_from_exp
        cfg = OmegaConf.load(cfg_path)
        cfg.trainer.devices = deviceconfig
        cfg.resume_from_exp = resume_from_exp

        # Overwrite the current .hydra directory with the one from the previous experiment
        current_hydra_dir = os.path.join(cfg.trainer.default_root_dir, '.hydra')
        previous_hydra_dir = os.path.join(resume_from_exp, '.hydra')
        shutil.copytree(previous_hydra_dir, current_hydra_dir,dirs_exist_ok=True)


    # Create datamodule
    dm = mixed_motion_dm.MixedMotionDataModule(cfg.data)

    # create model
    model = hydra.utils.instantiate(cfg.model)
    # model = torch.compile(model,fullgraph=True)
    # create dl logger
    logger = hydra.utils.instantiate(cfg.logger)
    

    # create trainer
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        every_n_epochs=cfg.trainer.check_val_every_n_epoch
    )
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, 
        callbacks=[checkpoint_callback],
        strategy=cfg.trainer.strategy,
        # strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    # torch.cuda.empty_cache()
    if hasattr(cfg, 'resume_from_exp'):
        ckptpath = sorted(
            glob.glob(os.path.join(cfg.resume_from_exp, 'tensorboard/version_0/checkpoints/*.ckpt')),
            key=os.path.getmtime
        )[-1]
        log.info(f"resume training from: <{ckptpath}>")
        trainer.fit(model = model, datamodule = dm,
                    ckpt_path=ckptpath)
        
    elif hasattr(cfg, 'finetune_from_exp'):
        ckptpath = sorted(
            glob.glob(os.path.join(cfg.finetune_from_exp, 'tensorboard/version_0/checkpoints/*.ckpt')),
            key=os.path.getmtime
        )[-1]
        log.info(f"fine tune from: <{ckptpath}>")
        
        #Load the checkpoint manually
        checkpoint = torch.load(ckptpath, map_location=lambda storage, loc: storage, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
            log.info(f"the EMA params are loaded from the basemodel.")
        
        trainer.fit(model = model, datamodule = dm)
    else:
        trainer.fit(model = model, datamodule = dm)
    

if __name__ == "__main__":
    train()