"""
Ray tune reporter that reports to wandb
"""
from ray.tune import CLIReporter

import wandb


class FlushingReporter(CLIReporter):

    def report(self, trials, done, *sys_info):
        wandb.run.summary['progress'] = self._progress_str(trials, done, *sys_info)
