"""
Helper method for reporting progress from tqdm to wandb
"""
import re

import wandb


class WandbFile:
    @staticmethod
    def write(text):
        percentage_string = re.search(r'\r +(\d\d?)%.*', text)
        if percentage_string is not None:
            wandb.log({'epoch_progress': int(percentage_string.group(1))})
        wandb.run.summary['progress_bar'] = text

    @staticmethod
    def flush():
        pass
