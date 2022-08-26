from inspect import getmembers, isfunction
from pathlib import Path
from types import FunctionType
from typing import List

import pandas as pd
from transformers import AutoTokenizer

from kbert.tokenizer import monkeypatches


class TMTokenizer(AutoTokenizer):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, index_files: List[Path] = None, *inputs, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        return cls.wrap(tokenizer)

    @classmethod
    def wrap(cls, tokenizer, index_files: List[Path] = None):
        tokenizer.token_index = pd.DataFrame(columns=['text', 'tokens', 'n_tokens'])

        tokenizer.batch_encode_plus_old = tokenizer.batch_encode_plus

        def add_monkeypatch(method_name):
            setattr(
                tokenizer,
                method_name,
                lambda *args, **kwargs: getattr(monkeypatches, method_name)(tokenizer, *args, **kwargs)
            )

        for method_name, obj in getmembers(monkeypatches, isfunction):
            if type(obj) == FunctionType and obj.__module__.endswith('.monkeypatches'):
                add_monkeypatch(method_name)

        if index_files is not None:
            for f in index_files:
                tokenizer.extend_index(f)

        return tokenizer
