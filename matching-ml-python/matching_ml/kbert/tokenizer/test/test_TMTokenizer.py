"""
Test for class TMTokenizer
"""
from pathlib import Path

from kbert.constants import RESOURCES_DIR, TRACK, TEST_CASE, TM_DATA_SPECIFICATION_PATH, FRACTION_SPECIFICATION, \
    MODEL_NAME
from utils import transformers_read_file, initialize_tokenizer


def test_tokenize():
    # Given
    root_dir = RESOURCES_DIR / Path('TM') / TRACK / TEST_CASE / TM_DATA_SPECIFICATION_PATH / FRACTION_SPECIFICATION
    data_left, data_right, labels = transformers_read_file(str(root_dir / 'train.csv'), True)
    max_length = 256
    tokenizer = initialize_tokenizer(True, MODEL_NAME, max_length, True, root_dir / 'index_train.csv')
    # When
    encodings = tokenizer(
        data_left[:128],
        # data_left,
        data_right[:128],
        data_right,
        return_tensors="pt",
        padding=True,
        truncation="longest_first",
        max_length=max_length,
    )

    # Then
    assert False
