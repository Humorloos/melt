import json
import pandas as pd
from pathlib import Path
from ray.tune import ExperimentAnalysis

from kbert.constants import ANALYSES_DIR

# %%

EXPERIMENT_DIR = '/ceph/lloos/melt/matching-ml-python/matching_ml/kbert/test/resources/ray_local_dir/2022-09-16_16.24'
anal = ExperimentAnalysis(EXPERIMENT_DIR, default_metric='f1', default_mode='max')
#%%


def save_metric_df(metric):
    metric_df = pd.DataFrame({int(k[-5:]): v[metric] for k, v in anal.trial_dataframes.items()})
    metric_df = metric_df.reindex(sorted(metric_df.columns), axis=1)
    target_dir = ANALYSES_DIR / 'fine_tuning_progress' / metric
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / 'TM.csv'
    metric_df.to_csv(target_path)


save_metric_df('f1')
save_metric_df('p')
save_metric_df('r')
