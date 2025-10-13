from .amass_smplx import *
import lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)




class MixedMotionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
        # create train test datasets
        if cfg.name == 'customized_action_mc':
            self.full_dataset = CustomizedActionMC(self.cfg)
        else:
            self.full_dataset = AMASS_SMPLX_NEUTRAL(self.cfg)
        
        # train val split. Note that val_dataset should not be empty, otherwise bugs in saving checkpoints.
        self.train_dataset, self.val_dataset = random_split(
                self.full_dataset, [self.cfg.train_test_split, 1.0-self.cfg.train_test_split])

        if self.val_dataset.__len__() == 0:
            self.val_dataset = self.train_dataset

        # define collate function
        self.collate_fn = collate_fn
        self.sampler=None

    def prepare_data(self):
        # check if data exist, else download
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.cfg.batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.cfg.batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          sampler=None)


class TestMotionDataModule(pl.LightningDataModule):
    """Unlike above, no train/val split, all sequences are used."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.data_dir = cfg.path
        self.cfg = cfg
        
        # create train test datasets
        if cfg.name=='amass_smplx':
            self.dataset = AMASS_SMPLX_NEUTRAL_TEST(self.cfg)
        else:
            raise NotImplementedError(f'Unknown dataset {cfg.name}')
        
    def prepare_data(self):
        # check if data exist, else download
        pass

    def setup(self, stage=None):
        # pass
        pass

    
    def dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=1,
                          num_workers=self.cfg.num_workers,
                          collate_fn=collate_fn)
