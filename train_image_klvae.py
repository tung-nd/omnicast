import os

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from weather_mae.data.image_datamodule import ImageDataModule
from weather_mae.models.klvae_module import KLVAEModule
from weather_mae.models.vaes.enc_dec_cnn import ResnetBlock
from weather_mae.utils.logging_utils import extract_latest_run_id

from lightning.pytorch.loggers.wandb import WandbLogger

class CustomCLI(LightningCLI):
    def _instantiate_trainer(self, config, callbacks):
        key = "callbacks"
        if key in config:
            if config[key] is None:
                config[key] = []
            elif not isinstance(config[key], list):
                config[key] = [config[key]]
            config[key].extend(callbacks)
            if key in self.trainer_defaults:
                value = self.trainer_defaults[key]
                config[key] += value if isinstance(value, list) else [value]
            if self.save_config_callback and not config.get("fast_dev_run", False):
                config_callback = self.save_config_callback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    **self.save_config_kwargs,
                )
                config[key].append(config_callback)
        else:
            rank_zero_warn(
                f"The `{self.trainer_class.__qualname__}` class does not expose the `{key}` argument so they will"
                " not be included."
            )
        
        if config['strategy'] == 'fsdp':
            fsdp_strategy = FSDPStrategy(
                sharding_strategy="SHARD_GRAD_OP",
                activation_checkpointing_policy={ResnetBlock},
                auto_wrap_policy={ResnetBlock}
            )
            config['strategy'] = fsdp_strategy
        
        return self.trainer_class(**config)

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = CustomCLI(
        model_class=KLVAEModule,
        datamodule_class=ImageDataModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)
    
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())

    logger_name = cli.trainer.logger._name
    for i in range(len(cli.trainer.callbacks)):
        if isinstance(cli.trainer.callbacks[i], ModelCheckpoint):
            cli.trainer.callbacks[i] = ModelCheckpoint(
                dirpath=os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints'),
                monitor=cli.trainer.callbacks[i].monitor,
                mode=cli.trainer.callbacks[i].mode,
                save_top_k=cli.trainer.callbacks[i].save_top_k,
                save_last=cli.trainer.callbacks[i].save_last,
                verbose=cli.trainer.callbacks[i].verbose,
                filename=cli.trainer.callbacks[i].filename,
                auto_insert_metric_name=cli.trainer.callbacks[i].auto_insert_metric_name
            )
            
    latest_run_id = extract_latest_run_id(cli.trainer.default_root_dir, logger_name)
    if latest_run_id is not None:
        print("Resuming wandb run with ID:", latest_run_id)
        cli.trainer.logger = WandbLogger(
            name=logger_name,
            project=cli.trainer.logger._wandb_init['project'],
            save_dir=os.path.join(cli.trainer.default_root_dir, logger_name),
            id=latest_run_id,
            resume='must'
        )
    else:
        cli.trainer.logger = WandbLogger(
            name=logger_name,
            project=cli.trainer.logger._wandb_init['project'],
            save_dir=os.path.join(cli.trainer.default_root_dir, logger_name)
        )

    if os.path.exists(os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints', 'last.ckpt')):
        ckpt_resume_path = os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints', 'last.ckpt')
    else:
        ckpt_resume_path = None

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_resume_path)


if __name__ == "__main__":
    main()
