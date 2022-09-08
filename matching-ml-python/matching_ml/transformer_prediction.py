import json
import logging
import os
import tempfile
from scipy.special import softmax

from kbert.models.sequence_classification.tmt_for_sequence_classification import PLTransformer
from utils import transformers_init, transformers_read_file, transformers_create_dataset, \
    transformers_get_training_arguments, initialize_tokenizer

log = logging.getLogger('matching_ml.python_server_melt')


def inner_transformers_prediction(request_headers):
    try:
        transformers_init(request_headers)

        model_name = request_headers["model-name"]
        prediction_file_path = request_headers["prediction-file-path"]
        tmp_dir = request_headers["tmp-dir"]
        using_tensorflow = request_headers["using-tf"].lower() == "true"
        change_class = request_headers["change-class"].lower() == "true"
        training_arguments = json.loads(request_headers["training-arguments"])

        from transformers import AutoTokenizer
        is_tm_modification_enabled = request_headers.get('tm', 'false').lower() == 'true'
        tm_attention = request_headers.get('tm-attention', 'false').lower() == 'true'
        tokenizer = initialize_tokenizer(is_tm_modification_enabled, model_name, int(request_headers['max-length']), training_arguments,
                                         tm_attention, request_headers["prediction-file-path"])

        log.info("Prepare transformers dataset and tokenize")
        data_left, data_right, _ = transformers_read_file(prediction_file_path, False)
        assert len(data_left) == len(data_right)
        predict_dataset = transformers_create_dataset(
            using_tensorflow, tokenizer, data_left, data_right
        )
        log.info("Transformers dataset contains %s rows.", len(data_left))

        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdirname:
            initial_arguments = {
                "report_to": "none",
                # 'disable_tqdm' : True,
            }
            fixed_arguments = {
                "output_dir": os.path.join(tmpdirname, "trainer_output_dir")
            }
            training_args = transformers_get_training_arguments(
                using_tensorflow, initial_arguments, training_arguments, fixed_arguments
            )

            log.info("Loading transformers model")
            if using_tensorflow:
                import tensorflow as tf

                log.info(
                    "Num gpu avail: " + str(len(tf.config.list_physical_devices("GPU")))
                )
                from transformers import TFTrainer, TFAutoModelForSequenceClassification

                with training_args.strategy.scope():
                    model = TFAutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=2
                    )

                trainer = TFTrainer(
                    model=model, tokenizer=tokenizer, args=training_args
                )
            else:
                import torch

                log.info("Is gpu used: " + str(torch.cuda.is_available()))
                from transformers import Trainer, AutoModelForSequenceClassification
                if is_tm_modification_enabled:
                    model = PLTransformer.from_pretrained(
                        model_name, num_labels=2, tm_attention=tm_attention
                    )
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=2
                    )

                trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args)

            log.info("Run prediction")
            pred_out = trainer.predict(predict_dataset)
            log.info(pred_out.metrics)
        class_index = 0 if change_class else 1
        # sigmoid: scores = 1 / (1 + np.exp(-pred_out.predictions, axis=1[:, class_index]))
        # compute softmax to get class probabilities (scores between 0 and 1)
        scores = softmax(pred_out.predictions, axis=1)[:, class_index]
        return scores.tolist()
    except Exception as e:
        import traceback

        return "ERROR " + traceback.format_exc()


