import torch
from einops import rearrange
from contextlib import contextmanager
from lightning import LightningModule
from weather_mae.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from weather_mae.models.vaes.continuous_vae_unet import KLVAEUnet
from weather_mae.models.ema import LitEma
from weather_mae.utils.forecast_metrics import lat_weighted_mse, lat_weighted_rmse
from weather_mae.utils.data_utils import WEIGHT_DICT


class KLVAEModule(LightningModule):
    def __init__(
        self,
        vae: KLVAEUnet,
        kl_weight: float = 1e-6,
        variable_weighting: bool = True,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vae"])
        
        self.vae = vae
        
        self.ema_decay = ema_decay
        if self.ema_decay:
            self.model_ema = LitEma(self, decay=ema_decay)
    
    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_decay:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema_decay:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def reconstruct(self, x, use_ema=False):
        if use_ema:
            with self.ema_scope():
                return self.vae(x)[0]
        else:
            return self.vae(x)[0]
    
    def loss(self, inputs, reconstructions, posterior, split, return_rmse=False):
        if len(inputs.shape) == 5:
            inputs = rearrange(inputs, 'b c t h w -> b t c h w').contiguous().flatten(0, 1)
        if len(reconstructions.shape) == 5:
            reconstructions = rearrange(reconstructions, 'b c t h w -> b t c h w').contiguous().flatten(0, 1)
        if self.hparams.variable_weighting:
            mse_loss_dict = lat_weighted_mse(reconstructions, inputs, self.trainer.datamodule.hparams.variables, self.lat, weighted=True, weight_dict=WEIGHT_DICT)
        else:
            mse_loss_dict = lat_weighted_mse(reconstructions, inputs, self.trainer.datamodule.hparams.variables, self.lat)
        rec_loss = mse_loss_dict['w_mse_agg']
        
        raw_recon = self.trainer.datamodule.denormalize(reconstructions)
        raw_input = self.trainer.datamodule.denormalize(inputs)
        rmse_loss_dict = lat_weighted_rmse(raw_recon, raw_input, self.trainer.datamodule.hparams.variables, self.lat)

        kl_loss = posterior.kl().mean()
        loss = rec_loss + self.hparams.kl_weight * kl_loss
        loss_dict = {
            f"{split}/gen_loss": loss.detach().item(),
            f"{split}/gen_kl_loss": kl_loss.detach().item(),
            f"{split}/gen_recon_loss": rec_loss.detach().item(),
        }
        if return_rmse:
            for k, v in rmse_loss_dict.items():
                loss_dict[f'{split}/gen_{k}'] = v.item()
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx): #, optimizer_idx
        inputs = batch
        reconstructions, posterior = self.vae(inputs)
        loss, loss_dict = self.loss(inputs, reconstructions, posterior, "train", return_rmse=False)
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=inputs.shape[0],
        )
        return loss
    
    def validation_step(self, batch, batch_idx): 
        self._validation_step(batch, batch_idx)
        if self.ema_decay:
            with self.ema_scope():
                self._validation_step(batch, batch_idx, suffix="_ema")

    def _validation_step(self, batch, batch_idx, suffix=""):
        inputs = batch
        reconstructions, posterior = self.vae(inputs)
        _, loss_dict = self.loss(inputs, reconstructions, posterior, "val" + suffix, return_rmse=True)
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=inputs.shape[0],
        )
        return loss_dict
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_decay:
            self.model_ema(self)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
            weight_decay=1e-5
        )

        n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs * n_steps,
            self.hparams.max_epochs * n_steps,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
