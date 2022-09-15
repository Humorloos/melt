from pathlib import Path

TM_DIR = Path(__file__).parent
MATCHING_ML_DIR = TM_DIR.parent
RESOURCES_DIR = TM_DIR / 'test' / 'resources'
PROJECT_DIR = MATCHING_ML_DIR.parent.parent.parent / 'master_thesis'
URI_PREFIX = "http://localhost:41193/"

PATIENCE = 4
MIN_DELTA = 0.001
MAX_VAL_SET_SIZE = 1024
MAX_EPOCHS = 5

# NUM_SAMPLES = 100
NUM_SAMPLES = 12  # as in hertling2021matching
WORKERS_PER_TRIAL = 12

DEBUG = False
# DEBUG = True
# RESUME = True  # True resumes at last checkpoint, False starts new trial
RESUME = False  # True resumes at last checkpoint, False starts new trial
RUN_NAME = '2022-09-09_20.31'
SEARCH_RESTORE_DIR_NAME = '2022-09-11_15.26'
# SEARCH_RESTORE_DIR_NAME = None

# GPU = [0, 1, 2, 3, 5, 6, 7]
GPU = [0, 2]
# GPU = [1]

# Independent variables of experiments
TM = True
TMA = True
REFERENCE_FRACTION = 1.0
MAX_LENGTH = 256
IS_MULTI = True

# For finetuning without HP optimization
USE_WEIGHTED_LOSS = False

TRACK = 'crosstrack'
TEST_CASE = "crosstestcase"

TM_DATA_DIR = RESOURCES_DIR / 'TM'
ORIGINAL_DATA_DIR = RESOURCES_DIR / 'original'
TM_TRACK_PATH = Path('TM') / TRACK
TM_TESTCASE_PATH = TM_TRACK_PATH / TEST_CASE
NORMALIZATION = 'normalized'  # normalized or raw
TM_DATA_SPECIFICATION_PATH = Path(NORMALIZATION) / 'all_targets' / f'isMulti_{str(IS_MULTI).lower()}'
TM_PATH = TM_TESTCASE_PATH / TM_DATA_SPECIFICATION_PATH
ORIGINAL_PATH = Path('original')
DATA_DIR = RESOURCES_DIR / {True: TM_PATH, False: ORIGINAL_PATH}[TM]

FRACTION_SPECIFICATION = f'posref{REFERENCE_FRACTION}'

TM_DATA_SPECIFICATION_PATH_WITH_FRACTION = TM_DATA_SPECIFICATION_PATH / FRACTION_SPECIFICATION
TM_PATH_WITH_FRACTION = TM_PATH / FRACTION_SPECIFICATION
DATA_DIR_WITH_FRACTION = RESOURCES_DIR / {True: TM_PATH_WITH_FRACTION, False: ORIGINAL_PATH}[TM]

TM_MODEL_PATH = Path('random') / 'mean_target' / f'maxLength_{MAX_LENGTH}' / f'tma_{TMA}' / 'isSwitch_false' / \
                'mouse-human-suite'
ORIGINAL_MODEL_PATH = '.'
MODEL_DIR = DATA_DIR_WITH_FRACTION / {True: TM_MODEL_PATH, False: ORIGINAL_MODEL_PATH}[TM] / \
            {True: "weighted_loss", False: "balanced_loss"}[USE_WEIGHTED_LOSS] / 'trained_model'

# Hyperparameters
# Original
# BATCH_SIZE = 64
# LR = 0.000003007
# DROPOUT_PROB = 0.1173
# POSITIVE_CLASS_WEIGHT = 0.7454
# WEIGHT_DECAY = 0.02477
# TM
BATCH_SIZE = 50
LR = 0.000005084
DROPOUT_PROB = 0.1513
POSITIVE_CLASS_WEIGHT = 0.8439
WEIGHT_DECAY = 0.08823

MODEL_NAME = "albert-base-v2"
CHECKPOINT_PATH = '/ceph/lloos/melt/matching-ml-python/matching_ml/kbert/test/resources/TM/normalized/all_targets/isMulti_true/posref1.0/random/mean_target/maxLength_256/tma_True/isSwitch_false/mouse-human-suite/balanced_loss/trained_model/master_thesis/8110eqzp/checkpoints/epoch=5-step=2936.ckpt'

DEFAULT_CONFIG = {
    'base_model': MODEL_NAME,
    'num_labels': 2,
    'tm': TM,
    'tm_attention': TM and TMA,
    'max_input_length': MAX_LENGTH,
    'gpus': ','.join([str(g) for g in GPU]),
    'dropout_prob': DROPOUT_PROB,
    'lr': LR,
    'weight_decay': WEIGHT_DECAY,
    'positive_class_weight': POSITIVE_CLASS_WEIGHT,
    'batch_size_train': BATCH_SIZE,
}
