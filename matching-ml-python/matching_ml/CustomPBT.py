"""
Custom implementation of ray tune's PBT, only for the purpose of logging
"""
import logging
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.trial import Trial
from typing import Dict, Optional, Callable

import wandb
from kbert.constants import TUNE_METRIC_MAPPING, DEBUG
from utils import initialize_wandb

logger = logging.getLogger(__name__)


class CustomPBT(PopulationBasedTraining):

    def __init__(self, time_attr: str = "time_total_s", metric: Optional[str] = None, mode: Optional[str] = None,
                 perturbation_interval: float = 60.0, burn_in_period: float = 0.0, hyperparam_mutations: Dict = None,
                 quantile_fraction: float = 0.25, resample_probability: float = 0.25,
                 custom_explore_fn: Optional[Callable] = None, log_config: bool = True, require_attrs: bool = True,
                 synch: bool = False, experiment_name=''):
        super().__init__(time_attr, metric, mode, perturbation_interval, burn_in_period, hyperparam_mutations,
                         quantile_fraction, resample_probability, custom_explore_fn, log_config, require_attrs, synch)

        if wandb.run is None and not DEBUG:
            initialize_wandb(experiment_name)

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict) -> str:
        wandb.log({f'{trial.trial_id[-2:]}_{k}': v for k, v in result.items() if k in TUNE_METRIC_MAPPING})
        return super().on_trial_result(trial_runner, trial, result)
