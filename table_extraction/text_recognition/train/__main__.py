import wandb
import pytorch_lightning as pl
import os
import torch
from pytorch_lightning.loggers import WandbLogger
from ..config import TextRecognitionConfig
from ..model import TextRecognitionModel
from ..datamodule import OCRDataModule
from ..util import get_training_parser, ModelCallback


def main():
    # PARSERs
    parser = get_training_parser()
    args = parser.parse_args()

    # CONFIG
    config = TextRecognitionConfig.from_args(args.model)
    config.update(args)
    # SEED
    pl.seed_everything(config.seed, workers=True)

    # WANDB (OPTIONAL)
    if args.wandb is not None:
        wandb.login(key=args.wandb)  # API KEY
        name = args.name or f"ocr-{config.max_epochs}-{config.batch_size}-{config.lr}"
        logger = WandbLogger(
            project="table-extraction",
            name=name,
            log_model=False
        )
    else:
        logger = None

    # DATAMODULE
    datamodule = OCRDataModule(config, data_dir=args.data_dir)
    # MODEL
    model = TextRecognitionModel.from_config(config)
    # CALLBACK
    root_path = os.path.join(os.getcwd(), "checkpoints")
    callback = ModelCallback(root_path=root_path)
    # STRATEGY
    strategy = 'ddp_find_unused_parameters_true' if torch.cuda.is_available() else 'auto'
    # TRAINER
    trainer = pl.Trainer(
        default_root_dir=root_path,
        logger=logger,
        callbacks=callback.get_callback(),
        gradient_clip_val=config.max_grad_norm,
        max_epochs=config.max_epochs,
        enable_progress_bar=args.pbar,
        deterministic=False,
        precision=args.precision,
        accumulate_grad_batches=max(config.max_batch_size // config.batch_size, 1),
        strategy=strategy
    )
    # FIT MODEL
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
