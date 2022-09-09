from pathlib import Path

TM_DIR = Path(__file__).parent
MATCHING_ML_DIR = TM_DIR.parent
RESOURCES_DIR = TM_DIR / 'test' / 'resources'
ORIGINAL_PATH = Path('original')
PROJECT_DIR = MATCHING_ML_DIR.parent.parent.parent / 'master_thesis'
URI_PREFIX = "http://localhost:41193/"

PATIENCE = 9
MIN_DELTA = 0.001
MAX_VAL_SET_SIZE = 512
MAX_EPOCHS = 5

NUM_SAMPLES = 100
WORKERS_PER_TRIAL = 12

DEBUG = False
# DEBUG = True
RESUME = True  # True resumes at last checkpoint, False starts new trial
# RESUME = False  # True resumes at last checkpoint, False starts new trial
RUN_NAME = '2022-09-09_20.31'

BATCH_SIZE = None
TM = False
TMA = True
MAX_LENGTH = 256
GPU = 4
USE_WEIGHTED_LOSS = False
TM_PATH = Path('TM') / 'normalized' / 'all_targets' / 'isMulti_true'
DATA_DIR = RESOURCES_DIR / {True: TM_PATH, False: ORIGINAL_PATH}[TM]
TM_MODEL_PATH = Path('random') / 'mean_target' / f'maxLength_{MAX_LENGTH}' / f'tma_{TMA}' / 'isSwitch_false' / \
                'mouse-human-suite'
ORIGINAL_MODEL_PATH = '.'
MODEL_DIR = DATA_DIR / {True: TM_MODEL_PATH, False: ORIGINAL_MODEL_PATH}[TM] / \
            {True: "weighted_loss", False: "balanced_loss"}[USE_WEIGHTED_LOSS] / 'trained_model'

DEFAULT_CONFIG = {
    'base_model': 'albert-base-v2',
    'num_labels': 2,
    'tm': TM,
    'tm_attention': TM and TMA,
    'max_input_length': MAX_LENGTH,
    'gpus': str(GPU),
    'dropout_prob': 0,
    'lr': 1e-5,
    'weight_decay': 1e-4,
    'positive_class_weight': 0.5,
    'batch_size_train': BATCH_SIZE,
}
