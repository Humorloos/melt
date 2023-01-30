"""
The functions find_max_batch_size was used to find max batch size for some time, but was not used in any experiments
"""
import logging
import numpy as np
import torch
from pytorch_lightning import Trainer

from MyDataModule import MyDataModule
from kbert.constants import DEFAULT_CONFIG
from kbert.models.sequence_classification.PLTransformer import PLTransformer
from utils import transformers_init, get_index_file_path

log = logging.getLogger('matching_ml.python_server_melt')


def find_max_batch_size(request_headers):
    try:
        return str(find_max_batch_size_(request_headers))

    except Exception as e:
        import traceback

        return "ERROR " + traceback.format_exc()


def find_max_batch_size_(request_headers):
    def is_header_true(header):
        return request_headers.get(header, 'false').lower() == 'true'

    transformers_init(request_headers)
    initial_model_name = request_headers["model-name"]
    training_file = request_headers["training-file"]
    tmp_dir = request_headers["tmp-dir"]

    is_tm_enabled = is_header_true('tm')
    is_tma_enabled = is_header_true('tm-attention')

    datamodule = MyDataModule(
        batch_size=1,
        train_data_path=training_file,
        num_workers=1,
        tm=is_tm_enabled,
        base_model=initial_model_name,
        tm_attention=is_tma_enabled,
        index_file_path=get_index_file_path(training_file)
    )
    datamodule.setup(stage='fit')
    # sort training dataset by lengths
    numpy_encoding_data = {key: value.detach().numpy() for key, value in datamodule.data_train.encodings.data.items()}
    input_lengths = (numpy_encoding_data['input_ids'] != 0).sum(1)
    len_order = np.flip(input_lengths.argsort())
    log.info("Loading transformers model")
    log.info("GPU used: " + str(torch.cuda.is_available()))
    model = PLTransformer.from_pretrained(DEFAULT_CONFIG)

    trainer_kwargs = {
        'max_steps': 1,
        'accelerator': 'gpu',
        'logger': False,
        'enable_progress_bar': False,
        'enable_checkpointing': False,
        'enable_model_summary': False,
        'num_sanity_val_steps': 0,
    }

    batch_size = 1
    step = 1
    largest_fitting_batch_size = 0
    while True:
        log.info("Run training")
        datamodule.data_train.encodings.data = {
            key: torch.Tensor(value[len_order][:batch_size]).long() for key, value in numpy_encoding_data.items()
        }
        datamodule.batch_size = batch_size
        try:
            trainer = Trainer(**trainer_kwargs)
            trainer.fit(model, datamodule=datamodule)
            log.info(f"Batch size of {batch_size} fits, trying larger value")
            largest_fitting_batch_size = batch_size
            batch_size += step
            step *= 2
        except RuntimeError as e:
            if e.args[0].startswith('CUDA out of memory'):
                log.info(f"Batch size of {batch_size} too large, will search for smaller value")
                step = largest_fitting_batch_size // 2
                batch_size -= step
                break
            else:
                raise e
    while True:
        step //= 2
        datamodule.data_train.encodings.data = {
            key: torch.Tensor(value[len_order][:batch_size]).long() for key, value in numpy_encoding_data.items()
        }
        datamodule.batch_size = batch_size
        try:
            trainer = Trainer(**trainer_kwargs)
            trainer.fit(model, datamodule=datamodule)
            log.info(f"Batch size of {batch_size} fits, increasing value by {step}")
            largest_fitting_batch_size = batch_size
            batch_size += step
        except RuntimeError as e:
            if e.args[0].startswith('CUDA out of memory'):
                log.info(f"Batch size of {batch_size} too large, drecreasing value by {step}")
                batch_size -= step
            else:
                raise e
        if step == 0:
            break
    size_ = largest_fitting_batch_size
    return size_
