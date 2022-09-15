import time
from contextlib import contextmanager
from transformers import AlbertModel

from kbert.models.sequence_classification.TMAlbertModel import TMAlbertModel
from kbert.monkeypatches import albert_forward, bert_get_extended_attention_mask


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


# currently only works for albert model, extend for supporting other models
def get_tm_variant(model):
    tm_albert = TMAlbertModel(model.albert.config)
    tm_albert.load_state_dict(model.albert.state_dict())
    model.albert = tm_albert
    return model
