import tempfile
from pathlib import Path
from transformers import BatchEncoding

from MyDatasetWithLabels import MyDatasetWithLabels
from kbert.models.sequence_classification.tmt_for_sequence_classification import TMTForSequenceClassification
from kbert.tokenizer.TMTokenizer import TMTokenizer
from utils import transformers_init, get_index_file_path, transformers_read_file, transformers_create_dataset, \
    transformers_get_training_arguments


def find_max_batch_size(app, request_headers):
    try:
        transformers_init(request_headers)

        initial_model_name = request_headers["model-name"]
        training_file = request_headers["training-file"]
        tmp_dir = request_headers["tmp-dir"]

        save_at_end = False

        from transformers import AutoTokenizer
        tokenizer = TMTokenizer.from_pretrained(initial_model_name, index_files=[get_index_file_path(training_file)])

        app.logger.info("Prepare transformers dataset and tokenize")
        data_left, data_right, labels = transformers_read_file(training_file, True)
        # j = 512
        # data_left = data_left[:j]
        # data_right = data_right[:j]
        # labels = labels[:j]
        assert len(data_left) == len(data_right) == len(labels)
        training_dataset = transformers_create_dataset(
            False, tokenizer, data_left, data_right, labels
        )

        # sort training dataset by lengths
        import numpy as np
        numpy_masks = training_dataset.encodings.data['attention_mask'].detach().numpy()
        input_lengths = numpy_masks.sum(1).max(1)
        len_order = np.flip(input_lengths.argsort())
        max_considered_batch_size = 256
        sorted_data_left = np.array(data_left)[len_order][:max_considered_batch_size]
        sorted_data_right = np.array(data_right)[len_order][:max_considered_batch_size]
        sorted_labels = np.array(labels)[len_order][:max_considered_batch_size]
        sorted_training_dataset = transformers_create_dataset(
            False, tokenizer, sorted_data_left.tolist(), sorted_data_right.tolist(), sorted_labels.tolist()
        )

        initial_arguments = {
            "report_to": "none",
        }

        app.logger.info("Loading transformers model")
        import torch

        app.logger.info("Using pytorch. GPU used: " + str(torch.cuda.is_available()))
        from transformers import Trainer, AutoModelForSequenceClassification
        model = TMTForSequenceClassification.from_pretrained(
            initial_model_name, num_labels=2
        )
        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdirname:
            fixed_arguments = {
                "output_dir": Path(tmpdirname) / "trainer_output_dir",
                "save_strategy": "no",
            }

            batch_size = 1
            step = 1
            largest_fitting_batch_size = 0

            while True:
                training_args = transformers_get_training_arguments(
                    False, initial_arguments, {'max_steps': 1, 'per_device_train_batch_size': batch_size},
                    fixed_arguments
                )

                app.logger.info(f'Batch size: {training_args.per_device_train_batch_size}')

                # tokenizer is added to the trainer because only in this case the tokenizer will be saved along the model to be reused.
                trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=MyDatasetWithLabels(
                        BatchEncoding(sorted_training_dataset[:batch_size]), labels[:batch_size]),
                    args=training_args,
                )

                app.logger.info("Run training")
                try:
                    trainer.train()
                    app.logger.info("Batch size fits, trying larger value")
                    largest_fitting_batch_size = batch_size
                    batch_size += step
                    step *= 2
                except RuntimeError as e:
                    if e.args[0].startswith('CUDA out of memory'):
                        app.logger.info(f"Batch size of {batch_size} too large, will search for smaller value")
                        step = largest_fitting_batch_size // 2
                        batch_size -= step
                        break
                    else:
                        raise e
                del trainer
                torch.cuda.empty_cache()

        while True:
            training_args = transformers_get_training_arguments(
                False, initial_arguments, {'max_steps': 1, 'per_device_train_batch_size': batch_size},
                fixed_arguments
            )

            app.logger.info(f'Batch size: {training_args.per_device_train_batch_size}')

            # tokenizer is added to the trainer because only in this case the tokenizer will be saved along the model to
            # be reused.
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=MyDatasetWithLabels(BatchEncoding(sorted_training_dataset[:batch_size]),
                                                  labels[:batch_size]),
                args=training_args,
            )

            app.logger.info("Run training")
            step //= 2
            try:
                trainer.train()
                app.logger.info(f"Batch size of {batch_size} fits, increasing value by {step}")
                largest_fitting_batch_size = batch_size
                batch_size += step
            except RuntimeError as e:
                if e.args[0].startswith('CUDA out of memory'):
                    app.logger.info(f"Batch size of {batch_size} too large, drecreasing value by {step}")
                    batch_size -= step
                else:
                    raise e
            del trainer
            torch.cuda.empty_cache()
            if step == 0:
                break

        return str(largest_fitting_batch_size)

    except Exception as e:
        import traceback

        return "ERROR " + traceback.format_exc()
