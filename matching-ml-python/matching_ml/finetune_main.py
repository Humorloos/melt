import logging
import sys

from kbert.constants import RESOURCES_DIR
from transformer_finetuning import finetune_transformer

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

BATCH_SIZE = 98
TM = True
TMA = True
MAX_LENGTH = 256
GPU = 4
USE_WEIGHTED_LOSS = True

DATA_DIR = RESOURCES_DIR / 'TM' / 'normalized' / 'all_targets' / 'isMulti_true'
MODEL_DIR = DATA_DIR / 'random' / 'mean_target' / f'maxLength_{MAX_LENGTH}' / f'tma_{TMA}' / 'isSwitch_false' / \
            'mouse-human-suite' / {True: "weighted_loss", False: "balanced_loss"}[USE_WEIGHTED_LOSS] / 'trained_model'

if __name__ == '__main__':
    pos_class_weight = {True: 2, False: -1.0}[USE_WEIGHTED_LOSS]
    result = finetune_transformer(
        request_headers={
            "model-name": "albert-base-v2",
            "using-tf": "false",
            "training-arguments": "{"
                                  f"\"per_device_train_batch_size\": {BATCH_SIZE},"
                                  f"\"weight_of_positive_class\": {pos_class_weight}"
                                  "}",
            "tmp-dir": str(RESOURCES_DIR),
            "tm": str(TM),
            'tm-attention': str(TMA),
            "multi-processing": "spawn",
            "resulting-model-location": str(MODEL_DIR),
            "training-file": str(DATA_DIR / 'train.csv'),
            "cuda-visible-devices": str(GPU),
            "max-length": str(MAX_LENGTH)
        },
    )
    print(result)
