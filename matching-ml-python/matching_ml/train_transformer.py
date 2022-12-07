import logging
import os
import pytorch_lightning as pl
import torch
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from ray import tune
from ray.tune.utils import wait_for_gpu
from torch import nn

import wandb
from CustomReportCheckpointCallback import CustomReportCheckpointCallback
from MyDataModule import MyDataModule
from kbert.constants import MIN_DELTA, PATIENCE, WORKERS_PER_TRIAL, DEBUG, MAX_EPOCHS, \
    TUNE_METRIC_MAPPING, MAX_GPU_UTIL
from kbert.models.sequence_classification.PLTransformer import PLTransformer
from utils import get_timestamp


def train_transformer(config, checkpoint_dir=None, do_tune=False):
    log = logging.getLogger('python_server_melt')
    # Fault tolerant training does not work with more than one worker unfortunately, see:
    # https://github.com/Lightning-AI/lightning/issues/12285
    # os.environ["PL_FAULT_TOLERANT_TRAINING"] = '1'
    if DEBUG:
        gpus = config.pop("gpus")
        if gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    elif do_tune:
        wait_for_gpu(target_util=MAX_GPU_UTIL, retry=2, delay_s=5)
    if do_tune:
        trial_name = tune.get_trial_name()
        wandb_logger = WandbLogger(
            name=trial_name, save_dir=None, id=trial_name,
            project="master_thesis", group=trial_name[:16], resume='allow')
        assert wandb_logger.experiment is not None
    save_dir = config.pop('save_dir')
    train_file = config.pop('data_dir')
    batch_size = int(config['batch_size'])

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
        # epoch = pl_load(checkpoint_path)['epoch']
        # When using PBT, we need to apply mutations after loading the model
        model.lr = config['lr']
        model.weight_decay = config['weight_decay']
        model.base_model.config.hidden_dropout_prob = config['dropout']
        model.loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0 - config['pos_weight'], config['pos_weight']])
        )

    else:
        log.info("Load new transformer model")
        print("Load new transformer model")
        model_path = Path(config['base_model'])
        if model_path.exists():
            model = PLTransformer.load_from_checkpoint(model_path)
        else:
            model = PLTransformer.from_pretrained(config)
        checkpoint_path = None
        # epoch = 0

    max_length = config['max_input_length']
    data_module = MyDataModule(
        # train_data_path=train_file,
        train_data_path=Path(train_file).parent / str(max_length) / 'train',
        batch_size=batch_size,
        num_workers={True: 1, False: WORKERS_PER_TRIAL}[DEBUG],
        tokenize=False,
        tm=config['tm'],
        base_model=config['base_model'],
        max_input_length=max_length,
        tm_attention=config['tm_attention'],
        soft_positioning=config['soft_positioning'],
        index_file_path=config['index_file_path'],
        one_epoch=do_tune,
    )
    log.info('Setup data module')
    data_module.setup(stage='fit', epoch=None, condensation_factor=config['condense'], pos_fraction=config['pos_frac'])

    # callbacks
    if do_tune:
        callbacks = [CustomReportCheckpointCallback(TUNE_METRIC_MAPPING, on='train_epoch_end')]
        wandb.log({k: v for k, v in config.items() if
                   k in {'pos_weight', 'condense', 'batch_size', 'lr', 'weight_decay', 'dropout'}})
    else:
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
        'accelerator': {True: 'gpu', False: 'cpu'}[torch.cuda.is_available()],
        'auto_select_gpus': True,
        'devices': 1,
        # 'val_check_interval': val_check_interval,
        'check_val_every_n_epoch': 1,
        'max_epochs': MAX_EPOCHS,
        'log_every_n_steps': 500 // batch_size,
    }
    if do_tune:
        trainer_kwargs.update({
            'enable_progress_bar': False,
            'enable_model_summary': False,
        })
    print('Load pytorch lightning Trainer')
    trainer = pl.Trainer(**trainer_kwargs)
    print("Run training")
    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
    return
