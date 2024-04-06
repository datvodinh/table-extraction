import wandb
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from text_recognition.config import SwinTransformerOCRConfig
from text_recognition.model import SwinTransformerOCR
from text_recognition.datamodule import OCRDataModule
from text_recognition.util import get_training_parser, ModelCallback


def main():
    # PARSERs
    parser = get_training_parser()
    args = parser.parse_args()

    # CONFIG
    config = SwinTransformerOCRConfig()
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
    model = SwinTransformerOCR(config)
    # CALLBACK
    root_path = os.path.join(os.getcwd(), "checkpoints")
    callback = ModelCallback(root_path=root_path)
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
        strategy="auto",
        use_distributed_sampler=False
    )
    # FIT MODEL
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
