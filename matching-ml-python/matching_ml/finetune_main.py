"""
This script provides an alternative entry-point to function finetune_transformer() which is usually called by the
flask-server for fine-tuning. Use this script to manually perform intermediate cross-track training.
"""
import logging
import sys

from kbert.constants import RESOURCES_DIR, BATCH_SIZE, TM, TMA, MAX_LENGTH, GPU, USE_WEIGHTED_LOSS, \
    DATA_DIR_WITH_FRACTION, MODEL_DIR, POSITIVE_CLASS_WEIGHT, DEBUG
from kbert.models.sequence_classification.find_max_batch_size import find_max_batch_size_
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

# TUNE = False
TUNE = True

if __name__ == '__main__':
    tmp_dir = str(RESOURCES_DIR)
    model_name = "albert-base-v2"
    training_file = str(DATA_DIR_WITH_FRACTION / 'train.csv')
    gpu = ','.join([str(g) for g in {True: [GPU[0]], False: GPU}[DEBUG]])
    max_length = str(MAX_LENGTH)
    tm_attention = str(TMA)
    shared_headers = {
        "model-name": model_name,
        "tmp-dir": tmp_dir,
        "training-file": training_file,
        "cuda-visible-devices": gpu,
        'max-length': max_length,
        'tm-attention': tm_attention,
        "tm": str(TM),
    }
    if BATCH_SIZE is None:
        batch_size = find_max_batch_size_(
            request_headers=shared_headers,
        )
    else:
        batch_size = BATCH_SIZE
    result = finetune_transformer(
        request_headers=shared_headers | {
            "resulting-model-location": str(MODEL_DIR),
            "training-arguments": "{"
                                  f"\"per_device_train_batch_size\": {batch_size},"
                                  f"\"weight_of_positive_class\": {POSITIVE_CLASS_WEIGHT}"
                                  "}",
            'tune': str(TUNE)
        },
    )
