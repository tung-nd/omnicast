import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from lightning import LightningDataModule

from weather_mae.data.iterative_dataset import ERA5ImageDataset


def collate_fn_with_filename(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, C, H, W
    filename = [batch[i][1] for i in range(len(batch))]
    return inp, filename


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        variables,
        return_filename=False,
        batch_size=1,
        val_batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # normalization for input
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)

        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)

        self.transforms = transforms.Normalize(normalize_mean, normalize_std)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon
    
    def get_transforms(self):
        return self.transforms
    
    def normalize(self, x):
        return self.transforms(x)
    
    def denormalize(self, x):
        return x * torch.from_numpy(self.transforms.std).to(x.device).reshape(1, -1, 1, 1) \
            + torch.from_numpy(self.transforms.mean).to(x.device).reshape(1, -1, 1, 1)

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ERA5ImageDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'train'),
                variables=self.hparams.variables,
                transform=self.transforms,
                return_filename=self.hparams.return_filename,
            )
            
            if os.path.exists(os.path.join(self.hparams.root_dir, 'val')):
                self.data_val = ERA5ImageDataset(
                    root_dir=os.path.join(self.hparams.root_dir, 'val'),
                    variables=self.hparams.variables,
                    transform=self.transforms,
                    return_filename=self.hparams.return_filename,
                )

            if os.path.exists(os.path.join(self.hparams.root_dir, 'test')):
                self.data_test = ERA5ImageDataset(
                    root_dir=os.path.join(self.hparams.root_dir, 'test'),
                    variables=self.hparams.variables,
                    transform=self.transforms,
                    return_filename=self.hparams.return_filename,
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_with_filename if self.hparams.return_filename else None,
        )

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=self.hparams.val_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn_with_filename if self.hparams.return_filename else None,
            )

    def test_dataloader(self):
        if self.data_test is not None:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.val_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn_with_filename if self.hparams.return_filename else None,
            )

# datamodule = VideoDataModule(
#     '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df',
#     variables=[
#         "2m_temperature",
#         "10m_u_component_of_wind",
#         "10m_v_component_of_wind",
#         "geopotential_500",
#         "temperature_850"
#     ],
#     steps=16,
#     interval=6,
#     data_freq=6,
#     batch_size=8,
#     val_batch_size=8,
#     num_workers=1,
#     pin_memory=False
# )
# datamodule.setup()
# for batch in datamodule.train_dataloader():
#     video = batch
#     print (video.shape)
#     break