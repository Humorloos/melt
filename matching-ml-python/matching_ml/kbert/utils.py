import pandas as pd
import time
import torch
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
from transformers import AlbertModel

from kbert.constants import TARGET_METRIC
from kbert.models.sequence_classification.TMAlbertModel import TMAlbertModel
from kbert.monkeypatches import albert_forward, bert_get_extended_attention_mask


@contextmanager
def print_time(description=''):
    start_time = time.time()
    if description != '':
        print(description)
    yield
    print(f'{description} took {time.time() - start_time} seconds')


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


def f_score(p, r, beta=2):
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
    return f


def get_metrics(prob, label, prefix, include_auc=False):
    # asdf = pd.DataFrame({k: v.detach().cpu().numpy() for k, v in {'label': label, 'prob': prob}.items()})
    tp = prob.dot(label)
    fp = prob.dot(1 - label)
    fn = (1 - prob).dot(label)
    r = tp / (tp + fn)
    p = tp / (tp + fp)
    f1 = f_score(p, r, 1)
    f2 = f_score(p, r, 2)
    metrics = {
        f'{prefix}_recall': r,
        f'{prefix}_precision': p,
        f'{prefix}_f1': f1,
        f'{prefix}_f2': f2
    }
    if include_auc:
        label = label.cpu().detach().numpy()
        if label.sum() > 0:
            metrics[f'{prefix}_auc'] = roc_auc_score(y_true=label, y_score=prob.cpu().detach().numpy())

    return metrics


def get_best_trial(analysis, metric=TARGET_METRIC):
    trial_results = pd.DataFrame([
        {metric: df[metric].max(), 'trial_id': df['trial_id'].iat[0]} for df in analysis.trial_dataframes.values()
    ])
    best_trial_id = trial_results.loc[trial_results[metric].idxmax(), 'trial_id']
    best_trial = next(t for t in analysis.trials if t.trial_id == best_trial_id)
    print(f'Best trial is {best_trial} with largest {metric} of {trial_results[metric].max()}')
    return best_trial
