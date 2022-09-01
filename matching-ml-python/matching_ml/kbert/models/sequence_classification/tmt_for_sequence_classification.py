from transformers import AutoModelForSequenceClassification

from kbert.utils import apply_tm_attention


class TMTForSequenceClassification(AutoModelForSequenceClassification):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        apply_tm_attention(model.albert)
        return model
