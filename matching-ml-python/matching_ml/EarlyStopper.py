from collections import defaultdict

import wandb
from ray.tune import Stopper
from typing import Dict


class EarlyStopper(Stopper):
    def __init__(self, patience, experiment_name, target_metric='f1'):
        self.target_metric = target_metric
        self.trials = defaultdict(lambda: {
            'results': [0],
            'iterations_since_last_improvement': 0,
            'iteration': 0,
        })
        self.iterations_since_last_improvement = 0
        self.patience = patience
        self.experiment_name = experiment_name
        self.wandb_initialized = False

    def __call__(self, trial_id: str, result: Dict):
        trial = self.trials[trial_id]
        largest_score = max(r for v in self.trials.values() for r in v['results'])
        current_score = result[self.target_metric]
        trial['results'].append(current_score)
        if current_score > largest_score:
            trial['iterations_since_last_improvement'] = 0
            for k, v in self.trials.items():
                if k != trial_id:
                    v['iterations_since_last_improvement'] = -1
            print(
                f'{self.target_metric} score of trial {trial_id} improved by {current_score - largest_score}. New best score: {current_score}.')
        else:
            smaller_stmt = f'{self.target_metric} score of trial {trial_id}: {current_score} < best score: {largest_score}.'
            trial['iterations_since_last_improvement'] += 1
            if trial['iterations_since_last_improvement'] > self.patience:
                print(f'{smaller_stmt} Score did not improve for {self.patience} iterations.')
            else:
                print(
                    f'{smaller_stmt} {self.patience - trial["iterations_since_last_improvement"]} iterations left before '
                    f'I will stop this trial.')
        if not self.wandb_initialized:
            wandb.init(
                project="master_thesis",
                name=self.experiment_name,
                id=self.experiment_name,
                group=self.experiment_name
            )
        trial_number = trial_id[-2:]
        wandb.log({
            f'{trial_number}_target_metric': current_score,
            f'{trial_number}_iteration': trial['iteration'],
            f'{trial_number}_space': self.patience - trial["iterations_since_last_improvement"],
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
