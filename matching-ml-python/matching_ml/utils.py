import csv
import logging
import os
import pathlib

from MyDatasetWithLabels import MyDatasetWithLabels

log = logging.getLogger('matching_ml.python_server_melt')


def transformers_init(request_headers):
    if "cuda-visible-devices" in request_headers:
        os.environ["CUDA_VISIBLE_DEVICES"] = request_headers["cuda-visible-devices"]

    if "transformers-cache" in request_headers:
        os.environ["TRANSFORMERS_CACHE"] = request_headers["transformers-cache"]


def get_index_file_path(corpus_file_path):
    my_path = pathlib.Path(corpus_file_path)
    return my_path.parent / f'index_{my_path.name}'


def transformers_read_file(file_path, with_labels):
    data_left = []
    data_right = []
    labels = []
    with open(file_path, encoding="utf-8") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            data_left.append(row[0])
            data_right.append(row[1])
            if with_labels:
                labels.append(int(row[2]))
    return data_left, data_right, labels


def transformers_create_dataset(
        using_tensorflow, tokenizer, left_sentences, right_sentences, labels=None
):
    tensor_type = "tf" if using_tensorflow else "pt"
    # padding (padding=True) is not applied here because the tokenizer is given to the trainer
    # which does the padding for each batch (more efficient)
    encodings = tokenizer(
        left_sentences,
        right_sentences,
        return_tensors=tensor_type,
        padding=True,
        truncation="longest_first",
    )

    if using_tensorflow:
        import tensorflow as tf

        if labels:
            return tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
        else:
            return tf.data.Dataset.from_tensor_slices(dict(encodings))
    else:
        import torch

        if labels:
            return MyDatasetWithLabels(encodings, labels)
        else:

            class MyDataset(torch.utils.data.Dataset):
                def __init__(self, encodings):
                    self.encodings = encodings

                def __getitem__(self, idx):
                    item = {
                        key: val[idx].detach().clone()
                        for key, val in self.encodings.items()
                    }
                    return item

                def __len__(self):
                    return len(self.encodings.input_ids)

            return MyDataset(encodings)


def transformers_get_training_arguments(
        using_tensorflow, initial_parameters, user_parameters, melt_parameters
):
    import dataclasses

    if using_tensorflow:
        from transformers import TFTrainingArguments

        allowed_arguments = set(
            [field.name for field in dataclasses.fields(TFTrainingArguments)]
        )
    else:
        from transformers import TrainingArguments

        allowed_arguments = set(
            [field.name for field in dataclasses.fields(TrainingArguments)]
        )

    training_arguments = dict(initial_parameters)
    training_arguments.update(user_parameters)
    training_arguments.update(melt_parameters)

    not_available = training_arguments.keys() - allowed_arguments
    if len(not_available) > 0:
        log.warning(
            "The following attributes are not set as training arguments because "
            + "they do not exist in the currently installed version of transformer: "
            + str(not_available)
        )
        for key_not_avail in not_available:
            del training_arguments[key_not_avail]
    if using_tensorflow:
        training_args = TFTrainingArguments(**training_arguments)
    else:
        training_args = TrainingArguments(**training_arguments)
    return training_args
