import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AlbertModel

from kbert.monkeypatches import albert_forward, pooling_forward, transformer_forward, bert_get_extended_attention_mask
from kbert.tokenizer.TMTokenizer import TMTokenizer

ROLE_RANKS = pd.Series({'s': 1, 'o': 0}, name='rank')


class KBertSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path, index_files=None, pooling_mode=None, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.tokenizer = TMTokenizer(self.tokenizer, index_files)

        transformer_module = self._first_module()
        self.max_seq_length = transformer_module.max_seq_length

        # KBert monkey patches
        transformer_module.forward = lambda features: transformer_forward(transformer_module, features)

        bert_model = transformer_module.auto_model
        bert_model.get_extended_attention_mask = \
            lambda attention_mask, input_shape: bert_get_extended_attention_mask(bert_model, attention_mask,
                                                                                 input_shape)
        if isinstance(bert_model, AlbertModel):
            bert_model.forward = \
                lambda input_ids, attention_mask, token_type_ids, position_ids, return_dict: albert_forward(
                    self=bert_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    return_dict=return_dict
                )

        pooling_module = self._last_module()
        pooling_module.forward = lambda features: pooling_forward(pooling_module, features)

        pooling_module.pooling_mode_first_target = False
        pooling_module.pooling_mode_mean_target = False
        if pooling_mode is not None and pooling_mode in {'first_target', 'mean_target'}:
            pooling_module.pooling_mode_mean_tokens = False
            pooling_module.pooling_mode_max_tokens = False
            pooling_module.pooling_mode_cls_token = False
            if pooling_mode == 'first_target':
                pooling_module.pooling_mode_first_target = True
            if pooling_mode == 'mean_target':
                pooling_module.pooling_mode_mean_target = True
