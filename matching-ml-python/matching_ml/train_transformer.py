import logging
import os
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from MyDataModule import MyDataModule
from kbert.constants import MATCHING_ML_DIR, DEBUG, MIN_DELTA, PATIENCE
from kbert.models.sequence_classification.PLTransformer import PLTransformer
from utils import get_timestamp


def train_transformer(config, checkpoint_dir=None, do_tune=False):
    log = logging.getLogger('python_server_melt')
    gpus = config.pop("gpus")
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    save_dir = config.pop('save_dir')
    train_file = config.pop('data_dir')
    val_check_interval = config.pop('val_check_interval')

    if checkpoint_dir is not None:
        log.info(f'Loading model from checkpoint {checkpoint_dir}')
        print(f'Loading model from checkpoint {checkpoint_dir}')
        # when using ray-tune and resuming training of a previously stopped model, load the model again from the
        # checkpoint provided by ray-tune
        model = PLTransformer.load_from_checkpoint(str(Path(checkpoint_dir) / "checkpoint"))
    else:
        log.info("Load new transformer model")
        print("Load new transformer model")
        model = PLTransformer.from_pretrained(config)
    # callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_f1',
            min_delta=MIN_DELTA,
            patience=PATIENCE,
            verbose=True,
            mode='max',
            # run check in each validation, not after training epoch
            check_on_train_epoch_end=False)
    ]
    if do_tune:
        callbacks.append(TuneReportCheckpointCallback(
            {'loss': 'val_loss', 'p': 'val_precision', 'r': 'val_recall', 'f1': 'val_f1'},
            on='validation_end'))
        wandb_logger = None
        # wandb_logger = WandbLogger(
        #     name=tune.get_trial_name(), save_dir=None, id=tune.get_trial_id(),
        #     project="master_thesis", log_model=True, config=config)
    else:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            name=get_timestamp(), save_dir=save_dir, project="master_thesis", log_model=True,
            config=config)
    trainer_kwargs = {
        'logger': wandb_logger,
        'callbacks': callbacks,
        'accelerator': 'gpu',
        'auto_select_gpus': True,
        'val_check_interval': val_check_interval
    }
    if do_tune:
        trainer_kwargs.update({
            'enable_progress_bar': False,
            'enable_model_summary': False,
        })
    print('Load pytorch lightning Trainer')
    with open(MATCHING_ML_DIR / 'log.txt', 'w') as fout:
        fout.write('forward')
    trainer = pl.Trainer(**trainer_kwargs)
    # results = trainer.train()
    print('Load tokenizer')
    datamodule = MyDataModule(
        train_data_path=train_file,
        batch_size=int(config['batch_size_train']),
        num_workers={True: 1, False: 12}[DEBUG],
        **config
    )
    print("Run training")
    with open(MATCHING_ML_DIR / 'log.txt', 'w') as fout:
        fout.write('forward')
    trainer.fit(model, datamodule=datamodule)
    # if save_at_end:
    #     log.info("Save model")
    #     trainer.save_model(config['save_dir'])
    return
