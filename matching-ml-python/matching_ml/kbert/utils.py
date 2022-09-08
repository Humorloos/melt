import pandas as pd
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from transformers import AlbertModel

from kbert.monkeypatches import bert_get_extended_attention_mask, albert_forward


@contextmanager
def print_time():
    start_time = time.time()
    yield
    print(f'took {time.time() - start_time} seconds')


def apply_tm_attention(transformer_model):
    transformer_model.get_extended_attention_mask = \
        lambda attention_mask, input_shape: bert_get_extended_attention_mask(
            transformer_model, attention_mask, input_shape)
    if isinstance(transformer_model, AlbertModel):
        transformer_model.forward = \
            lambda *args, **kwargs: albert_forward(
                self=transformer_model,
                *args,
                **kwargs,
            )


def get_timestamp():
    return pd.Timestamp.today(
        tz=datetime.now(timezone(timedelta(0))).astimezone().tzinfo
    ).strftime('%Y-%m-%d_%H.%M')
