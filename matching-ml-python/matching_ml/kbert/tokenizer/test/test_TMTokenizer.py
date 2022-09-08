from kbert.KBertSentenceTransformer import KBertSentenceTransformer
from kbert.constants import RESOURCES_DIR
from utils import transformers_read_file


def test_tokenize():
    # Given
    model_dir = RESOURCES_DIR / 'kbert' / 'normalized' / 'all_targets' / 'isMulti_false'
    data_left, data_right, labels = transformers_read_file(str(model_dir / 'train.csv'), True)
    tokenizer = KBertSentenceTransformer('paraphrase-albert-small-v2', [model_dir / 'index_train.csv']).tokenizer
    # When
    encodings = tokenizer(
        data_left[:128],
        # data_left,
        data_right[:128],
        data_right,
        return_tensors="pt",
        padding=True,
        truncation="longest_first",
        max_length=38,
    )

    # Then
    assert False
