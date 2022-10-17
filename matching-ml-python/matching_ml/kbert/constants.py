from pathlib import Path

TM_DIR = Path(__file__).parent
MATCHING_ML_DIR = TM_DIR.parent
RESOURCES_DIR = TM_DIR / 'test' / 'resources'
PROJECT_DIR = MATCHING_ML_DIR.parent.parent.parent / 'master_thesis'
ANALYSES_DIR = PROJECT_DIR / 'analyses'
URI_PREFIX = "http://localhost:41193/"

PATIENCE = 10
MIN_DELTA = 0.001
MAX_VAL_SET_SIZE = 2 ** 12
# MAX_VAL_SET_SIZE = 2 ** 10
MAX_EPOCH_EXAMPLES = 2 ** 14
# MAX_EPOCH_EXAMPLES = 2 ** 11
MAX_EPOCHS = 1000

# NUM_SAMPLES = 3
NUM_SAMPLES = 12  # as in hertling2021matching
# WORKERS_PER_TRIAL = 1
WORKERS_PER_TRIAL = 12

DEBUG = False
# DEBUG = True
RESUME = False  # True resumes at last checkpoint, False starts new trial
# RESUME = True  # True resumes at last checkpoint, False starts new trial
# RUN_NAME = '2022-09-25_16.29'  # TM 5% f2
# RUN_NAME = '2022-09-26_15.29'  # original 5% f2
# RUN_NAME = '2022-09-27_09.17'  # original 5% f1
# RUN_NAME = '2022-09-30_17.47'  # original 20% f1
RUN_NAME = '2022-10-14_16.27'

# from scheduler.utils.utils import get_free_memory_by_gpu
GPU = [0, 1, 2, 3, 4, 5, 6, 7]  # get_free_memory_by_gpu()
# GPU = [0, 3, 4, 5, 6, 7]
# GPU = [0, 2, 4, 3]
# GPU = [0, 5]
# GPU = [6]
# GPUS_PER_TRIAL = 0.5  # 0.5 trains 2 models per GPU
GPUS_PER_TRIAL = 1  # 0.5 trains 2 models per GPU

# Independent variables of experiments
# TM = True
# TMA = True
TM = False
TMA = False
REFERENCE_FRACTION = 0.2
# REFERENCE_FRACTION = 0.05
# REFERENCE_FRACTION = 1.0
ALL_NEGATIVES = True
# ALL_NEGATIVES = False
MAX_LENGTH = 256
IS_MULTI = True
TARGET_METRIC = 'f1'

BATCH_SIZE = 32
# For finetuning without HP optimization
USE_WEIGHTED_LOSS = False

TRACK = 'opal_1_1'
TEST_CASE = "crosstestcase"

TM_DATA_DIR = RESOURCES_DIR / 'TM'
ORIGINAL_DATA_DIR = RESOURCES_DIR / 'original'
TRACK_PATH = Path({True: 'TM', False: 'original'}[TM]) / TRACK
TESTCASE_PATH = TRACK_PATH / TEST_CASE
NORMALIZATION = 'normalized'  # normalized or raw
TM_DATA_SPECIFICATION_PATH = Path(NORMALIZATION) / 'all_targets' / f'isMulti_{str(IS_MULTI).lower()}'
TM_PATH = TESTCASE_PATH / TM_DATA_SPECIFICATION_PATH
DATA_DIR = RESOURCES_DIR / {True: TM_PATH, False: TESTCASE_PATH}[TM]
ALL_NEGATIVES_SPECIFICATION = {True: '_all_negatives', False: ''}[ALL_NEGATIVES]
FRACTION_SPECIFICATION = f'posref{REFERENCE_FRACTION}{ALL_NEGATIVES_SPECIFICATION}'

TM_DATA_SPECIFICATION_PATH_WITH_FRACTION = TM_DATA_SPECIFICATION_PATH / FRACTION_SPECIFICATION
DATA_DIR_WITH_FRACTION = RESOURCES_DIR / {True: TM_PATH, False: TESTCASE_PATH}[TM] / FRACTION_SPECIFICATION

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
LR = 0.000005084
DROPOUT_PROB = 0.1513
POSITIVE_CLASS_WEIGHT = 0.8439
WEIGHT_DECAY = 0.08823

MODEL_NAME = "albert-base-v2"
CHECKPOINT_PATH = '/ceph/lloos/master_thesis/melt-target/ftTrack/anatomy_track/TextExtractorSet/trained_model_checkpoint/checkpoint'

DEFAULT_CONFIG = {
    'base_model': MODEL_NAME,
    'num_labels': 2,
    'tm': TM,
    'tm_attention': TM and TMA,
    'max_input_length': MAX_LENGTH,
    'gpus': ','.join([str(g) for g in GPU]),
    'dropout': DROPOUT_PROB,
    'lr': LR,
    'weight_decay': WEIGHT_DECAY,
    'pos_weight': POSITIVE_CLASS_WEIGHT,
    'batch_size': BATCH_SIZE,
    'condense': 14,
}

TUNE_METRIC_MAPPING = {
    'loss': 'val_loss',
    'p': 'val_precision',
    'r': 'val_recall',
    'f1': 'val_f1',
    'f2': 'val_f2',
    'auc': 'val_auc',
    'bin_p': 'val_bin_precision',
    'bin_r': 'val_bin_recall',
    'bin_f1': 'val_bin_f1',
    'bin_f2': 'val_bin_f2',
    'epoch_time': 'epoch_time',
    'validation_time': 'validation_time',
}
