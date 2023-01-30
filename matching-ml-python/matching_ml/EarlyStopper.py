"""
Custom implementation of early stopping criterion for PBT populations
"""
from collections import defaultdict

from pathlib import Path
from ray.tune import Stopper
from typing import Dict

import wandb
from kbert.constants import DEBUG
from utils import initialize_wandb


class EarlyStopper(Stopper):
    def __init__(self, patience, min_delta, experiment_name, ray_local_dir, target_metric='f1'):
        self.target_metric = target_metric
        self.trials = defaultdict(lambda: {
            'results': [0],
            'iterations_since_last_improvement': 0,
            'iteration': 0,
        })
        self.iterations_since_last_improvement = 0
        self.patience = patience
        self.min_delta = min_delta
        self.experiment_name = experiment_name
        self.wandb_initialized = False
        self.experiment_dir = Path(ray_local_dir) / self.experiment_name
        self.best_checkpoint = None

    def __call__(self, trial_id: str, result: Dict):
        trial = self.trials[trial_id]
        largest_score = max(r for v in self.trials.values() for r in v['results'])
        current_score = result[self.target_metric]
        trial['results'].append(current_score)
        delta = current_score - largest_score
        if delta >= self.min_delta:
            trial['iterations_since_last_improvement'] = 0
            for k, v in self.trials.items():
                if k != trial_id:
                    v['iterations_since_last_improvement'] = -1

            # Add trial checkpoint as best checkpoint so far
            trial_dir = self.experiment_dir / f'{self.experiment_name}_{trial_id}'
            trial_checkpoint_paths = [
                c
                for p in trial_dir.iterdir() if p.name.startswith('checkpoint')
                for c in p.iterdir() if c.name == 'checkpoint'
            ]
            trial_checkpoint_modification_times = {c: c.stat().st_mtime_ns for c in trial_checkpoint_paths}
            most_recent_modification_time = max(trial_checkpoint_modification_times.values())
            self.best_checkpoint = next(
                c for c, t in trial_checkpoint_modification_times.items() if t == most_recent_modification_time
            )
            print(
                f'{self.target_metric} score of trial {trial_id} improved by {delta} > {self.min_delta}. '
                f'New best score: {current_score}. New best checkpoint: {self.best_checkpoint.absolute()}'
            )
        else:
            smaller_stmt = f'{self.target_metric} score of trial {trial_id}: {current_score}, ' \
                           f'best score: {largest_score}, delta: {delta} < {self.min_delta}.'
            trial['iterations_since_last_improvement'] += 1
            if trial['iterations_since_last_improvement'] > self.patience:
                print(f'{smaller_stmt} Score did not improve for {self.patience} iterations.')
            else:
                print(
                    f'{smaller_stmt} {self.patience - trial["iterations_since_last_improvement"]} '
                    f'iterations left before I will stop this trial.'
                )
        if not DEBUG:
            if not self.wandb_initialized:
                if wandb.run is None:
                    initialize_wandb(self.experiment_name)
                self.wandb_initialized = True
            trial_number = trial_id[-2:]
            wandb.log({
                f'{trial_number}_iteration': trial['iteration'],
                f'{trial_number}_space': self.patience - trial["iterations_since_last_improvement"],
                'best_checkpoint': str(self.best_checkpoint.absolute()),
            })
        trial['iteration'] += 1
        return False

    def stop_all(self):
        if len(self.trials) == 0:
            return False
        has_converged = all(i['iterations_since_last_improvement'] > self.patience for i in self.trials.values())
        if has_converged:
            print(f'No trial has improved for {self.patience} iterations. Stopping experiment.')
        return has_converged
