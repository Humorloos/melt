import logging
import sys
from pathlib import Path
from pytorch_lightning import Trainer
from ray.tune import ExperimentAnalysis

from MyDataModule import MyDataModule
from kbert.constants import RESOURCES_DIR, BATCH_SIZE, TM, TMA, GPU, DATA_DIR_WITH_FRACTION, MODEL_DIR, \
    POSITIVE_CLASS_WEIGHT, RUN_NAME, WORKERS_PER_TRIAL, TARGET_METRIC, MODEL_NAME
from kbert.models.sequence_classification.PLTransformer import PLTransformer
from kbert.models.sequence_classification.find_max_batch_size import find_max_batch_size_
from kbert.utils import get_best_trial
from transformer_finetuning import finetune_transformer
from utils import transformers_init
#%%

logging.basicConfig(
    handlers=[
        logging.FileHandler(__file__ + ".log", "a+", "utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    format="%(asctime)s %(levelname)-5s ExternalPythonProcess     - %(message)s",
    level=logging.INFO,
)
logging.addLevelName(logging.WARNING, "WARN")
logging.addLevelName(logging.CRITICAL, "FATAL")

if __name__ == '__main__':
    transformers_init({'cuda-visible-devices': ','.join(str(g) for g in GPU)})

    model_name = MODEL_NAME
    test_file_name = 'test_14.csv'
    test_file = str(DATA_DIR_WITH_FRACTION / test_file_name)
    index_file_path = DATA_DIR_WITH_FRACTION / f'index_{test_file_name}'

    analysis = ExperimentAnalysis(RESOURCES_DIR / 'ray_local_dir' / RUN_NAME)

    best_trial = get_best_trial(analysis)
    model = PLTransformer.load_from_checkpoint(
        Path(analysis.get_best_checkpoint(best_trial, TARGET_METRIC, 'max')) / 'checkpoint')
    tm = model.hparams.config['tm']
    base_model = model.base_model.name_or_path
    max_input_length = model.hparams.config['max_input_length']
    tm_attention = model.hparams.config['tm_attention']
    datamodule = MyDataModule(test_data_path=test_file, num_workers=WORKERS_PER_TRIAL, tm=tm, base_model=base_model,
                              max_input_length=max_input_length, tm_attention=tm_attention,
                              index_file_path=index_file_path)
    trainer_kwargs = {
        'accelerator': 'gpu',
        'auto_select_gpus': True,
        'max_epochs': -1,
        'logger': False,
        'devices': 1
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.test(model, datamodule=datamodule)
