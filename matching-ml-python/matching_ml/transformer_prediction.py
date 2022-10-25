import os

import json
import logging
import torch
from pytorch_lightning import Trainer
from scipy.special import softmax

from MyDataModule import MyDataModule
from kbert.constants import DEBUG
from kbert.models.sequence_classification.PLTransformer import PLTransformer
from utils import transformers_init, get_index_file_path

log = logging.getLogger('matching_ml.python_server_melt')


def inner_transformers_prediction(request_headers):
    try:
        return transformer_predict(request_headers)
    except Exception as e:
        import traceback

        return "ERROR " + traceback.format_exc()


def transformer_predict(request_headers):
    transformers_init(request_headers)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"][0]
    prediction_file_path = request_headers["prediction-file-path"]
    change_class = request_headers["change-class"].lower() == "true"
    training_arguments = json.loads(request_headers["training-arguments"])
    is_tm_modification_enabled = request_headers.get('tm', 'false').lower() == 'true'
    tm_attention = request_headers.get('tm-attention', 'false').lower() == 'true'
    max_length = int(request_headers['max-length'])
    tma_text = {True: ' and TM attention mask', False: ' but without attention mask'}[tm_attention]
    tm_text = {
        True: f'with TM modification{tma_text}. Max molecule length is {max_length}',
        False: 'without TM modification'
    }[is_tm_modification_enabled]
    log.info(f"Loading transformers model {tm_text}")
    checkpoint_path = request_headers['model-name']
    model = PLTransformer.load_from_checkpoint(checkpoint_path)
    model_name = model.base_model.name_or_path
    datamodule = MyDataModule(
        predict_data_path=prediction_file_path,
        num_workers={True: 1, False: 12}[DEBUG],
        tm=is_tm_modification_enabled,
        base_model=model_name,
        max_input_length=max_length,
        tm_attention=tm_attention,
        index_file_path=training_arguments.get('index_file', get_index_file_path(prediction_file_path))
    )
    trainer_kwargs = {
        'accelerator': 'gpu',
        'auto_select_gpus': True,
        'max_epochs': -1,
        'logger': False,
    }
    trainer = Trainer(**trainer_kwargs)

    log.info("Is gpu used: " + str(torch.cuda.is_available()))
    log.info("Run prediction")
    pred_out = trainer.predict(model, datamodule=datamodule)
    class_index = 0 if change_class else 1
    # sigmoid: scores = 1 / (1 + np.exp(-pred_out.predictions, axis=1[:, class_index]))
    # compute softmax to get class probabilities (scores between 0 and 1)
    scores = softmax(torch.concat([p.logits for p in pred_out]), axis=1)[:, class_index]
    return scores.tolist()
