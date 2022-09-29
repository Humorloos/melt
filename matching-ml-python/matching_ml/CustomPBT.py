import logging
import random
import wandb
from ray.tune import TuneError
from ray.tune.checkpoint_manager import _TuneCheckpoint
from ray.tune.schedulers import PopulationBasedTraining, TrialScheduler
from ray.tune.schedulers.pbt import make_experiment_tag
from ray.tune.trial import Trial
from ray.util import log_once
from typing import List, Dict, Optional, Callable

from PBTHistoricalState import PBTHistoricalState
from kbert.constants import TUNE_METRIC_MAPPING
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
        self.best_trial_states = []

        if wandb.run is None:
            initialize_wandb(experiment_name)

    def custom_checkpoint_or_exploit(self, trial: Trial, trial_executor: "trial_runner.RayTrialExecutor",
                                     upper_quantile: List[Trial], lower_quantile: List[Trial], score):
        """Checkpoint if in upper quantile, exploits if in lower."""
        state = self._trial_state[trial]
        if trial in upper_quantile:
            # The trial last result is only updated after the scheduler
            # callback. So, we override with the current result.
            logger.debug("Trial {} is in upper quantile".format(trial))
            logger.debug("Checkpointing {}".format(trial))
            if trial.status == Trial.PAUSED:
                # Paused trial will always have an in-memory checkpoint.
                checkpoint = trial.checkpoint
                state.last_checkpoint = checkpoint
            else:
                checkpoint = trial_executor.save(trial, _TuneCheckpoint.MEMORY, result=state.last_result)
                state.last_checkpoint = checkpoint
            if len(self.best_trial_states) > 0:
                worst_trial_state = min(self.best_trial_states)
            else:
                worst_trial_state = PBTHistoricalState(score=0, checkpoint=None)
            if score > worst_trial_state.score:
                if len(self.best_trial_states) == len(upper_quantile):
                    self.best_trial_states.remove(worst_trial_state)
                self.best_trial_states.append(PBTHistoricalState(score=score, checkpoint=checkpoint))
            self._num_checkpoints += 1
        else:
            state.last_checkpoint = None  # not a top trial

        if trial in lower_quantile:
            logger.debug("Trial {} is in lower quantile".format(trial))
            trial_to_clone = random.choice(upper_quantile)
            assert trial is not trial_to_clone
            if not self._trial_state[trial_to_clone].last_checkpoint:
                logger.info(
                    "[pbt]: no checkpoint for trial."
                    " Skip exploit for Trial {}".format(trial)
                )
                return
            self._exploit(trial_executor, trial, trial_to_clone)

    def _exploit(
            self,
            trial_executor: "trial_executor.TrialExecutor",
            trial: Trial,
            trial_to_clone: Trial,
    ):
        """Transfers perturbed state from trial_to_clone -> trial.

        If specified, also logs the updated hyperparam state.
        """
        trial_state = self._trial_state[trial]
        new_state = self._trial_state[trial_to_clone]
        logger.info(
            "[exploit] transferring weights from trial "
            "{} (score {}) -> {} (score {})".format(
                trial_to_clone, new_state.last_score, trial, trial_state.last_score
            )
        )

        new_config = self._get_new_config(trial, trial_to_clone)

        # Only log mutated hyperparameters and not entire config.
        old_hparams = {
            k: v
            for k, v in trial_to_clone.config.items()
            if k in self._hyperparam_mutations
        }
        new_hparams = {
            k: v for k, v in new_config.items() if k in self._hyperparam_mutations
        }
        logger.info(
            "[explore] perturbed config from {} -> {}".format(old_hparams, new_hparams)
        )

        if self._log_config:
            self._log_config_on_step(
                trial_state, new_state, trial, trial_to_clone, new_config
            )

        new_tag = make_experiment_tag(
            trial_state.orig_tag, new_config, self._hyperparam_mutations
        )
        if trial.status == Trial.PAUSED:
            # If trial is paused we update it with a new checkpoint.
            # When the trial is started again, the new checkpoint is used.
            if not self._synch:
                raise TuneError(
                    "Trials should be paused here only if in "
                    "synchronous mode. If you encounter this error"
                    " please raise an issue on Ray Github."
                )
        else:
            trial_executor.stop_trial(trial)
            trial_executor.set_status(trial, Trial.PAUSED)
        trial.set_experiment_tag(new_tag)
        trial.set_config(new_config)
        hist_state = random.choice(self.best_trial_states)
        trial.on_checkpoint(hist_state.checkpoint)

        self._num_perturbations += 1
        # Transfer over the last perturbation time as well
        trial_state.last_perturbation_time = new_state.last_perturbation_time
        trial_state.last_train_time = new_state.last_train_time

    def on_trial_result(
            self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ) -> str:
        if self._time_attr not in result:
            time_missing_msg = (
                "Cannot find time_attr {} "
                "in trial result {}. Make sure that this "
                "attribute is returned in the "
                "results of your Trainable.".format(self._time_attr, result)
            )
            if self._require_attrs:
                raise RuntimeError(
                    time_missing_msg
                    + "If this error is expected, you can change this to "
                      "a warning message by "
                      "setting PBT(require_attrs=False)"
                )
            else:
                if log_once("pbt-time_attr-error"):
                    logger.warning(time_missing_msg)
        if self._metric not in result:
            metric_missing_msg = (
                "Cannot find metric {} in trial result {}. "
                "Make sure that this attribute is returned "
                "in the "
                "results of your Trainable.".format(self._metric, result)
            )
            if self._require_attrs:
                raise RuntimeError(
                    metric_missing_msg + "If this error is expected, "
                                         "you can change this to a warning message by "
                                         "setting PBT(require_attrs=False)"
                )
            else:
                if log_once("pbt-metric-error"):
                    logger.warning(metric_missing_msg)

        wandb.log({f'{trial.trial_id[-2:]}_{k}': v for k, v in result.items() if k in TUNE_METRIC_MAPPING})

        if self._metric not in result or self._time_attr not in result:
            return TrialScheduler.CONTINUE

        time = result[self._time_attr]
        state = self._trial_state[trial]

        # Continue training if burn-in period has not been reached, yet.
        if time < self._burn_in_period:
            return TrialScheduler.CONTINUE

        # Continue training if perturbation interval has not been reached, yet.
        if time - state.last_perturbation_time < self._perturbation_interval:
            return TrialScheduler.CONTINUE  # avoid checkpoint overhead

        score = self._save_trial_state(state, time, result, trial)

        if not self._synch:
            state.last_perturbation_time = time
            lower_quantile, upper_quantile = self._quantiles()
            decision = TrialScheduler.CONTINUE
            for other_trial in trial_runner.get_trials():
                if other_trial.status in [Trial.PENDING, Trial.PAUSED]:
                    decision = TrialScheduler.PAUSE
                    break
            self.custom_checkpoint_or_exploit(
                trial, trial_runner.trial_executor, upper_quantile, lower_quantile, score
            )
            return TrialScheduler.NOOP if trial.status == Trial.PAUSED else decision
        else:
            # Synchronous mode.
            if any(
                    self._trial_state[t].last_train_time < self._next_perturbation_sync
                    and t != trial
                    for t in trial_runner.get_trials()
            ):
                logger.debug("Pausing trial {}".format(trial))
            else:
                # All trials are synced at the same timestep.
                lower_quantile, upper_quantile = self._quantiles()
                all_trials = trial_runner.get_trials()
                not_in_quantile = []
                for t in all_trials:
                    if t not in lower_quantile and t not in upper_quantile:
                        not_in_quantile.append(t)
                # Move upper quantile trials to beginning and lower quantile
                # to end. This ensures that checkpointing of strong trials
                # occurs before exploiting of weaker ones.
                all_trials = upper_quantile + not_in_quantile + lower_quantile
                for t in all_trials:
                    logger.debug("Perturbing Trial {}".format(t))
                    self._trial_state[t].last_perturbation_time = time
                    self._checkpoint_or_exploit(
                        t, trial_runner.trial_executor, upper_quantile, lower_quantile
                    )

                all_train_times = [
                    self._trial_state[t].last_train_time
                    for t in trial_runner.get_trials()
                ]
                max_last_train_time = max(all_train_times)
                self._next_perturbation_sync = max(
                    self._next_perturbation_sync + self._perturbation_interval,
                    max_last_train_time,
                )
            # In sync mode we should pause all trials once result comes in.
            # Once a perturbation step happens for all trials, they should
            # still all be paused.
            # choose_trial_to_run will then pick the next trial to run out of
            # the paused trials.
            return (
                TrialScheduler.NOOP
                if trial.status == Trial.PAUSED
                else TrialScheduler.PAUSE
            )
