from typing import Optional

from dataclasses import dataclass
from functools import total_ordering
from ray.tune.checkpoint_manager import _TuneCheckpoint


@total_ordering
@dataclass
class PBTHistoricalState:
    score: int = 0
    checkpoint: Optional[_TuneCheckpoint] = None

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return other.score == self.score
