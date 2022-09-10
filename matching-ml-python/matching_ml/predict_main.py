import logging
import sys

from kbert.constants import RESOURCES_DIR, TM, TMA, MAX_LENGTH, GPU, DATA_DIR, CHECKPOINT_PATH
from transformer_prediction import transformer_predict

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
    tmp_dir = str(RESOURCES_DIR)
    prediction_file = str(DATA_DIR / 'predict.csv')
    gpu = str(GPU)
    max_length = str(MAX_LENGTH)
    tm_attention = str(TMA)
    shared_headers = {
        "model-name": CHECKPOINT_PATH,
        "tmp-dir": tmp_dir,
        'prediction-file-path': prediction_file,
        "cuda-visible-devices": gpu,
        'max-length': max_length,
        'tm-attention': tm_attention,
        "tm": str(TM),
        'change-class': 'False',
        'training-arguments': '{}'
    }
    result = transformer_predict(
        request_headers=shared_headers,
    )
