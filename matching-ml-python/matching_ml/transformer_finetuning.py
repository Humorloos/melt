import json
import logging
import numpy as np
import pytorch_lightning as pl
import tempfile
import torch
from math import ceil
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight

from MyDataModuleWithLabels import MyDataModuleWithLabels
from kbert.constants import MIN_DELTA, PATIENCE
from kbert.models.sequence_classification.tmt_for_sequence_classification import PLTransformer
from kbert.utils import get_timestamp
from utils import transformers_init, initialize_tokenizer, transformers_read_file

log = logging.getLogger('python_server_melt')


def inner_transformers_finetuning(request_headers):
    try:
        finetune_transformer(request_headers)
        return "True"
    except Exception as e:
        import traceback

        return "ERROR " + traceback.format_exc()


def finetune_transformer(request_headers):
    transformers_init(request_headers)
    initial_model_name = request_headers["model-name"]
    resulting_model_location = request_headers["resulting-model-location"]
    tmp_dir = request_headers["tmp-dir"]
    training_file = request_headers["training-file"]
    using_tensorflow = request_headers["using-tf"].lower() == "true"
    training_arguments = json.loads(request_headers["training-arguments"])
    save_at_end = training_arguments.get("save_at_end", True)
    training_arguments.pop("save_at_end", None)  # delete if existent
    weight_of_positive_class = training_arguments.get("weight_of_positive_class", -1.0)
    training_arguments.pop("weight_of_positive_class", None)  # delete if existent
    is_tm_modification_enabled = request_headers.get('tm', 'false').lower() == 'true'
    tm_attention = is_tm_modification_enabled and request_headers.get('tm-attention', 'false').lower() == 'true'
    max_input_length = int(request_headers['max-length'])
    config = {
        'base_model': initial_model_name,
        'num_labels': 2,
        'batch_size_train': training_arguments['per_device_train_batch_size'],
        'tm': is_tm_modification_enabled,
        'tm_attention': tm_attention,
        'max_input_length': max_input_length,
        'positive_class_weight': weight_of_positive_class
    }

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdirname:
        # initial_arguments = {
        #     "report_to": "none",
        #     # 'disable_tqdm' : True,
        # }
        #
        # fixed_arguments = {
        #     "output_dir": os.path.join(tmpdirname, "trainer_output_dir"),
        #     "save_strategy": "no",
        # }
        #
        # training_args = transformers_get_training_arguments(
        #     using_tensorflow, initial_arguments, training_arguments, fixed_arguments
        # )
        batch_size = training_arguments["per_device_train_batch_size"]
        log.info(f'Batch size: {batch_size}')
        data_left, data_right, labels = transformers_read_file(training_file, True)

        log.info("GPU used: " + str(torch.cuda.is_available()))
        log.info("Loading transformers model")
        class_weights = [0.5, 0.5]  # only one label available -> default to [0.5, 0.5]

        if weight_of_positive_class >= 0.0:
            # calculate class weights
            if weight_of_positive_class > 1.0:
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
            else:
                class_weights = [1.0 - weight_of_positive_class, weight_of_positive_class]
            log.info("Using class weights: " + str(class_weights))
        model = PLTransformer.from_pretrained({'class_weights': class_weights},
                                              initial_model_name, num_labels=2, tm_attention=tm_attention
                                              )

                # trainer = WeightedLossTrainer(
                #     model=model,
                #     tokenizer=tokenizer,
                #     train_dataset=training_dataset,
                #     compute_metrics=compute_metrics,
                #     args=training_args,
                # )
                # trainer.set_melt_weight(class_weights)
            # else:
            #     log.info("Using standard loss")
            #     # tokenizer is added to the trainer because only in this case the tokenizer will be saved along the model to be reused.
            #     trainer = Trainer(
            #         model=model,
            #         tokenizer=tokenizer,
            #         train_dataset=training_dataset,
            #         args=training_args,
            #         compute_metrics=compute_metrics,
            #     )

        # callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                min_delta=MIN_DELTA,
                patience=PATIENCE,
                verbose=True,
                # run check in each validation, not after training epoch
                check_on_train_epoch_end=False)
        ]
        wandb_logger = WandbLogger(
            name=get_timestamp(), save_dir=resulting_model_location, project="master_thesis", log_model=True,
            config=config)

        # determine val check interval (spend 10 times more time on training than on validation)
        complete_data_size = len(labels)

        val_set_size = min(1000, complete_data_size // 10)  # size of validation set
        train_set_size = complete_data_size - val_set_size  # size of training set
        val_check_interval = 1 / (ceil(1 / (10 * val_set_size / train_set_size)))

        trainer_kwargs = {
            'logger': wandb_logger,
            'callbacks': callbacks,
            'devices': 1,
            'accelerator': 'gpu',
            'auto_select_gpus': True,
            'max_epochs': 99,
            'val_check_interval': val_check_interval
        }
        trainer = pl.Trainer(**trainer_kwargs)

        # results = trainer.train()
        log.info('Load tokenizer')
        tokenizer = initialize_tokenizer(is_tm_modification_enabled, initial_model_name, max_input_length,
                                         training_arguments, tm_attention, request_headers["training-file"])
        datamodule = MyDataModuleWithLabels(training_file, batch_size=batch_size, tokenizer=tokenizer)
        log.info("Run training")
        results = trainer.fit(model, datamodule=datamodule)

        # if save_at_end:
        #     log.info("Save model")
        #     trainer.save_model(resulting_model_location)
    return results
