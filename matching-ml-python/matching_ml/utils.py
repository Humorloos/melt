import logging
import numpy as np
import os
import pandas as pd
import pathlib
from datetime import datetime, timezone, timedelta
from math import ceil
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer

import wandb
from MyDatasetWithLabels import MyDatasetWithLabels
from kbert.tokenizer.TMTokenizer import TMTokenizer
from kbert.utils import print_time

log = logging.getLogger('matching_ml.python_server_melt')


def transformers_init(request_headers):
    if "cuda-visible-devices" in request_headers:
        os.environ["CUDA_VISIBLE_DEVICES"] = request_headers["cuda-visible-devices"]

    if "transformers-cache" in request_headers:
        os.environ["TRANSFORMERS_CACHE"] = request_headers["transformers-cache"]

    # see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'


def get_index_file_path(corpus_file_path):
    my_path = pathlib.Path(corpus_file_path)
    return my_path.parent / f'index_{my_path.name}'


def transformers_read_file(file_path, with_labels):
    df = transformers_get_df(file_path, with_labels)
    labels = []
    if with_labels:
        labels = df['label'].tolist()
    return df['text_left'].tolist(), df['text_right'].tolist(), labels


def transformers_get_df(file_path, with_labels=True):
    with print_time(f'loading df {file_path}'):
        csv = pd.read_csv(file_path, names=['text_left', 'text_right'] + {True: ['label'], False: []}[with_labels])
    return csv


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


def initialize_tokenizer(is_tm_modification_enabled, model_name, max_length, tm_attention, index_file_path=None):
    log.info('load tokenizer')
    with print_time('loading tokenizer'):
        if is_tm_modification_enabled:
            tokenizer = TMTokenizer.from_pretrained(
                model_name, index_files=[index_file_path],
                max_length=max_length,
                tm_attention=tm_attention
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    return compute_metrics_inner(labels, preds)


def compute_metrics_inner(labels, preds):
    preds_binary = preds.argmax(-1)
    acc = accuracy_score(labels, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds_binary, average="binary", pos_label=1, zero_division=0
    )
    preds_proba = softmax(preds, axis=1)[:, 1]
    auc = roc_auc_score(labels, preds_proba)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "aucf1": auc + f1,
    }


def get_trial_name(trial):
    """Function for generating trial names"""
    return f"{get_timestamp()}_{trial.trial_id}"


def get_timestamp():
    return pd.Timestamp.today(
        tz=datetime.now(timezone(timedelta(0))).astimezone().tzinfo
    ).strftime('%Y-%m-%d_%H.%M')


def initialize_wandb(name):
    wandb.init(
        project="master_thesis",
        name=name,
        id=name,
        group=name
    )


def load_fragmented_df(dir, min_pos, min_neg, random_state: np.random.RandomState = None) -> pd.DataFrame:
    label_df = pd.DataFrame([{
        'required_splits': ceil(n / 500),
        'dir': dir / f'label_{i}'
    } for i, n in enumerate([min_neg, min_pos])])
    label_df['n_splits'] = label_df['dir'].apply(lambda d: len(list(iter(d.iterdir()))))
    if random_state is None:
        choice = np.random.choice
    else:
        choice = random_state.choice
    label_df['split_ids'] = label_df[['required_splits', 'n_splits']].apply(
        lambda row: choice(row['n_splits'], row['required_splits']), axis=1
    )
    label_df['df'] = label_df[['dir', 'split_ids']].apply(
        lambda row: load_fragmented_df_for_label(row, random_state, min_neg, min_pos),
        axis=1
    )
    all_dfs = label_df['df'].values
    merged_df = pd.concat(all_dfs, ignore_index=True)
    padded_df = merged_df.apply(pad_to_max_len)
    return padded_df.sample(frac=1, random_state=random_state)


def load_fragmented_df_for_label(row, random_state, min_neg, min_pos):
    df = pd.concat([pd.read_pickle(row['dir'] / f'split_{i}.pickle') for i in row["split_ids"]], ignore_index=True)
    sample_size = min([min_neg, min_pos][row.name], df.shape[0])
    return df.sample(sample_size, random_state=random_state)


def pad_to_max_len(col: pd.Series):
    first_element = col.iat[0]
    if not isinstance(first_element, np.ndarray):
        return col
    dim = len(first_element.shape)
    lengths: pd.Series = col.apply(len)
    max_len = lengths.max()
    if (lengths == max_len).all():
        return col
    if dim == 1:
        return col.apply(lambda a: np.pad(a, (0, max_len - a.shape[0])))
    elif dim == 2:
        return col.apply(lambda a: np.pad(a, (0, max_len - a.shape[0], 0, max_len - a.shape[0])))
