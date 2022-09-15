# This unit test checks the python_server.py
# Run `pytest` in the root directory of the jRDF2vec project (where the pom.xml resides).
import logging
import pytest
import requests
import sys
from pathlib import Path
from scipy.special import softmax
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer

from kbert.constants import RESOURCES_DIR, URI_PREFIX
from kbert.models.sequence_classification.PLTransformer import PLTransformer
from kbert.test.ServerThread import ServerThread
from kbert.tokenizer.TMTokenizer import TMTokenizer
from python_server_melt import app as my_app
from transformer_finetuning import finetune_transformer
from utils import transformers_init, transformers_read_file, transformers_create_dataset, compute_metrics

logging.basicConfig(
    handlers=[
        logging.FileHandler(__file__ + ".log", "a+", "utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    # format="PythonServer: %(asctime)s %(levelname)s:%(message)s",
    format="%(asctime)s %(levelname)-5s ExternalPythonProcess     - %(message)s",
    level=logging.INFO,
)
logging.addLevelName(logging.WARNING, "WARN")
logging.addLevelName(logging.CRITICAL, "FATAL")


@pytest.fixture
def server_thread():
    server_thread = ServerThread()
    # wait_time_seconds = 10
    server_thread.start()
    # print(f"Waiting {wait_time_seconds} seconds for the server to start.")
    # time.sleep(wait_time_seconds)
    yield
    print("Shutting down...")
    server_thread.stop()


@pytest.fixture()
def app():
    my_app.config.update({
        "TESTING": True,
    })
    yield my_app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_get_vector(server_thread):
    test_model_vectors = "../../test/resources/test_model_vectors.kv"
    vector_test_path = Path(test_model_vectors)
    assert vector_test_path.is_file()
    result = requests.get(
        URI_PREFIX + "get-vector",
        headers={"concept": "Europe", "vector_path": test_model_vectors},
    )
    assert len(result.content.decode("utf-8").split(" ")) == 100


def test_is_in_vocabulary(server_thread):
    test_model = "../../test/resources/test_model"
    test_vectors = "../../test/resources/test_model_vectors.kv"
    model_test_path = Path(test_model)
    vector_test_path = Path(test_vectors)
    assert model_test_path.is_file()
    assert vector_test_path.is_file()
    result = requests.get(
        URI_PREFIX + "is-in-vocabulary",
        headers={"concept": "Europe", "model_path": test_model},
    )
    assert result.content.decode("utf-8") == "True"
    result = requests.get(
        URI_PREFIX + "is-in-vocabulary",
        headers={"concept": "Europe", "vector_path": test_vectors},
    )
    assert result.content.decode("utf-8") == "True"


def test_get_similarity(server_thread):
    test_model = "../../test/resources/test_model"
    model_test_path = Path(test_model)
    assert model_test_path.is_file()
    result = requests.get(
        URI_PREFIX + "get-similarity",
        headers={
            "concept_1": "Europe",
            "concept_2": "united",
            "model_path": test_model,
        },
    )
    result_str = result.content.decode("utf-8")
    assert float(result_str) > 0


def test_sentence_transformers_prediction_kbert(client):
    # def test_sentence_transformers_prediction_kbert():
    test_model = 'paraphrase-albert-small-v2'
    response = client.get(
        "/sentencetransformers-prediction",
        headers={
            "pooling-mode": "mean_target",
            "sampling-mode": "stratified",
            "tm": "true",
            "model-name": "paraphrase-albert-small-v2",
            "using-tf": "false",
            "training-arguments": "{}",
            "tmp-dir": str(RESOURCES_DIR),
            "multi-processing": "no_multi_process",
            "corpus-file-name": str(RESOURCES_DIR / 'kbert' / 'raw' / 'all_targets' / 'corpus.csv'),
            "queries-file-name": str(RESOURCES_DIR / 'kbert' / 'raw' / 'all_targets' / 'queries.csv'),
            "query-chunk-size": "100",
            "corpus-chunk-size": "500000",
            "topk": "5",
            "both-directions": "true",
            "topk-per-resource": "true",
        },
    )
    print('')


def test_sentence_transformers_prediction(client):
    # def test_sentence_transformers_prediction_kbert():
    response = client.get(
        "/sentencetransformers-prediction",
        headers={
            "model-name": "paraphrase-albert-small-v2",
            "using-tf": "false",
            "training-arguments": "{}",
            "tmp-dir": str(RESOURCES_DIR),
            "multi-processing": "no_multi_process",
            "corpus-file-name": str(RESOURCES_DIR / 'original' / 'corpus.txt'),
            "queries-file-name": str(RESOURCES_DIR / 'original' / 'queries.txt'),
            "query-chunk-size": "100",
            "corpus-chunk-size": "500000",
            "topk": "5",
            "both-directions": "true",
            "topk-per-resource": "true",
        },
    )
    print('')


def test_transformers_finetuning_tm(client):
    model_dir = RESOURCES_DIR / 'kbert' / 'normalized' / 'all_targets' / 'isMulti_true'
    response = client.get(
        "/transformers-finetuning",
        headers={
            "model-name": "albert-base-v2",
            "using-tf": "false",
            "training-arguments": "{"
                                  "\"per_device_train_batch_size\": 98,"
                                  "\"weight_of_positive_class\": -1.0"
                                  "}",
            "tmp-dir": str(RESOURCES_DIR),
            "tm": "true",
            "multi-processing": "spawn",
            "resulting-model-location": str(model_dir / 'trained_model'),
            "training-file": str(model_dir / 'train.csv'),
            "cuda-visible-devices": '4',
            "max-length": '256'
        },
    )
    print(response.data)


def test_finetune_tm():
    model_dir = RESOURCES_DIR / 'kbert' / 'normalized' / 'all_targets' / 'isMulti_true'
    log = logging.getLogger('python_server_melt')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log2 = logging.getLogger('transformers.trainer')
    log2.setLevel(logging.INFO)
    result = finetune_transformer(
        request_headers={
            "model-name": "albert-base-v2",
            "using-tf": "false",
            "training-arguments": "{"
                                  "\"per_device_train_batch_size\": 98,"
                                  "\"weight_of_positive_class\": -1.0"
                                  "}",
            "tmp-dir": str(RESOURCES_DIR),
            "tm": "true",
            "multi-processing": "spawn",
            "resulting-model-location": str(model_dir / 'trained_model'),
            "training-file": str(model_dir / 'train.csv'),
            "cuda-visible-devices": '4',
            "max-length": '256'
        },
    )
    print(result)


def test_transformers_finetuning(client):
    # def test_sentence_transformers_prediction_kbert():
    model_dir = RESOURCES_DIR / 'original'
    response = client.get(
        "/transformers-finetuning",
        headers={
            "model-name": "albert-base-v2",
            "using-tf": "false",
            "training-arguments": "{}",
            "tmp-dir": str(RESOURCES_DIR),
            "tm": "false",
            "multi-processing": "spawn",
            "resulting-model-location": str(model_dir / 'trained_model'),
            "training-file": str(model_dir / 'train.txt'),
            "cuda-visible-devices": '4'
        },
    )
    print('')


def test_transformers_predict(client):
    model_dir = RESOURCES_DIR / 'original'

    response = client.get('/transformers-prediction', headers={
        'model-name': str(model_dir / 'trained_model'),
        'using-tf': 'false',
        'training-arguments': '{}',
        'tmp-dir': str(RESOURCES_DIR),
        'multi-processing': 'spawn',
        'tm': 'false',
        'prediction-file-path': str(model_dir / 'predict.txt'),
        'change-class': 'false'})
    print('')


def test_evaluate(client):
    transformers_init({
        "cuda-visible-devices": '4'
    })
    # def test_sentence_transformers_prediction_kbert():
    model_dir = RESOURCES_DIR / 'original'
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / 'trained_model'), num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / 'trained_model'))
    data_left, data_right, labels = transformers_read_file(str(model_dir / 'predict.txt'), True)
    dataset = transformers_create_dataset(
        False, tokenizer, data_left[:256], data_right[:256], labels[:256]
        # False, tokenizer, data_left, data_right, labels
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    asdf = trainer.predict(dataset)
    scores = softmax(asdf.predictions, axis=1)[:, 0]

    print('scores')


def test_evaluate_tm(client):
    TMA = True
    MAX_LENGTH = 256
    USE_WEIGHTED_LOSS = False

    SAMPLE_SIZE = 512

    data_path = 'TM/normalized/all_targets/isMulti_true'
    DATA_DIR = RESOURCES_DIR / data_path
    # melt_trained_models_root = PROJECT_DIR / 'melt-target/ftTestCase/albert-base-v2/posHighPrecisionMatcher/TM/' / \
    #                            data_path
    model_dir = DATA_DIR / 'random' / 'mean_target' / f'maxLength_{MAX_LENGTH}' / f'tma_{TMA}' / 'isSwitch_false' / \
                'mouse-human-suite' / {True: "weighted_loss", False: "balanced_loss"}[
                    USE_WEIGHTED_LOSS] / 'trained_model'
    transformers_init({
        "cuda-visible-devices": '4'
    })

    model = PLTransformer.from_pretrained(model_dir, num_labels=2)
    tokenizer = TMTokenizer.from_pretrained(model_dir, index_files=[DATA_DIR / 'index_train.csv'],
                                            max_length=MAX_LENGTH)
    data_left, data_right, labels = transformers_read_file(str(DATA_DIR / 'train.csv'), True)
    dataset = transformers_create_dataset(
        False, tokenizer, data_left[:SAMPLE_SIZE], data_right[:SAMPLE_SIZE], labels[:SAMPLE_SIZE]
        # False, tokenizer, data_left, data_right, labels
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    asdf = trainer.predict(dataset)
    scores = softmax(asdf.predictions, axis=1)[:, 0]

    print('scores')


def test_transformers_finetuning_tm_profile(client):
    model_dir = RESOURCES_DIR / 'kbert' / 'normalized' / 'all_targets' / 'isMulti_true'
    response = client.get(
        "/transformers-finetuning",
        headers={
            "model-name": "albert-base-v2",
            "using-tf": "false",
            "training-arguments": "{"
                                  "\"per_device_train_batch_size\": 41, "
                                  "\"max_steps\": 1, "
                                  "\"save_at_end\": \"false\""
                                  "}",
            "tmp-dir": str(RESOURCES_DIR),
            "tm": "true",
            "multi-processing": "spawn",
            "resulting-model-location": str(model_dir / 'trained_model'),
            "training-file": str(model_dir / 'train.csv'),
            "cuda-visible-devices": '4'
        },
    )
    print('')


def test_find_max_batch_size(client):
    model_dir = RESOURCES_DIR / 'kbert' / 'normalized' / 'all_targets' / 'isMulti_true'
    response = client.get(
        '/tm-find-max-batch-size',
        headers={
            "model-name": "albert-base-v2",
            "tmp-dir": str(RESOURCES_DIR),
            "training-file": str(model_dir / 'train.csv'),
            "cuda-visible-devices": '4',
            'max-length': '256',
            'tm-attention': 'true',
        },
    )
    print('')
