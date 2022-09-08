import torch
from transformers import Trainer


class WeightedLossTrainer(Trainer):

    def set_melt_weight(self, melt_weight_arg):
        self.melt_weight = torch.FloatTensor(melt_weight_arg).to(device=self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.melt_weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
