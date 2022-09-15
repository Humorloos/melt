import logging
import os
import pytorch_lightning as pl
import torch
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from torch import nn

from MyDataModule import MyDataModule
from kbert.constants import MATCHING_ML_DIR, MIN_DELTA, PATIENCE, WORKERS_PER_TRIAL, DEBUG, MAX_EPOCHS
from kbert.models.sequence_classification.PLTransformer import PLTransformer
from kbert.tokenizer.constants import SEED
from utils import get_timestamp


def train_transformer(config, checkpoint_dir=None, do_tune=False):
    pl.seed_everything(SEED)
    log = logging.getLogger('python_server_melt')
    # gpus = config.pop("gpus")
    # if gpus is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    save_dir = config.pop('save_dir')
    train_file = config.pop('data_dir')
    val_check_interval = config.pop('val_check_interval')
    data_module = MyDataModule(
        train_data_path=train_file,
        batch_size=int(config['batch_size_train']),
        num_workers={True: 1, False: WORKERS_PER_TRIAL}[DEBUG],
        tm=config['tm'],
        base_model=config['base_model'],
        max_input_length=config['max_input_length'],
        tm_attention=config['tm_attention'],
        index_file_path=config['index_file_path']
    )
    data_module.setup(stage='fit')

    # 'lr': tune.loguniform(1e-7, 1e-5),
    # 'weight_decay': tune.loguniform(1e-7, 1e-1),
    # 'dropout_prob': tune.uniform(0.1, 0.5),
    # 'positive_class_weight': tune.uniform(0.75, 0.9)
    if checkpoint_dir is not None:
        log.info(f'Loading model from checkpoint {checkpoint_dir}')
        print(f'Loading model from checkpoint {checkpoint_dir}')
        # when using ray-tune and resuming training of a previously stopped model, load the model again from the
        # checkpoint provided by ray-tune
        if do_tune:
            checkpoint_path = str(Path(checkpoint_dir) / "checkpoint")
        else:
            checkpoint_path = checkpoint_dir
        model = PLTransformer.load_from_checkpoint(checkpoint_path)
        # When using PBT, we need to apply mutations after loading the model
        model.lr = config['lr']
        model.weight_decay = config['weight_decay']
        model.base_model.config.hidden_dropout_prob = config['dropout_prob']
        model.loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0 - config['positive_class_weight'], config['positive_class_weight']])
        )

    else:
        log.info("Load new transformer model")
        print("Load new transformer model")
        model = PLTransformer.from_pretrained(config)
        checkpoint_path = None
    # callbacks
    if do_tune:
        callbacks = [
            TuneReportCheckpointCallback(
                {'loss': 'val_loss', 'p': 'val_precision', 'r': 'val_recall', 'f1': 'val_f1'},
                on='validation_end'
            )
        ]
        wandb_logger = None
    else:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            name=get_timestamp(), save_dir=save_dir, project="master_thesis", log_model=True,
            config=config)
        callbacks = [
            EarlyStopping(
                monitor='val_f1',
                min_delta=MIN_DELTA,
                patience=PATIENCE,
                verbose=True,
                mode='max',
                # run check in each validation, not after training epoch
                check_on_train_epoch_end=False
            ),
            ModelCheckpoint(
                monitor='val_f1',
                mode='max',
            ),
        ]
    trainer_kwargs = {
        'logger': wandb_logger,
        'callbacks': callbacks,
        'accelerator': 'gpu',
        'auto_select_gpus': True,
        'devices': 1,
        'val_check_interval': val_check_interval,
        'max_epochs': MAX_EPOCHS,
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
    print("Run training")
    with open(MATCHING_ML_DIR / 'log.txt', 'w') as fout:
        fout.write('forward')
    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
    return
