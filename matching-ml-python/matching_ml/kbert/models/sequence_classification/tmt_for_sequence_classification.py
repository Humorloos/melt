import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from kbert.utils import apply_tm_attention


class PLTransformer(pl.LightningModule):
    @classmethod
    def from_pretrained(cls, config, pretrained_model_name_or_path, tm_attention=True, *model_args, **kwargs):
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        if tm_attention:
            apply_tm_attention(model.albert)
        config['model'] = model
        return PLTransformer(config)

    def __init__(self, config):
        super().__init__()
        self.base_model = config['model']

        self.a = torchmetrics.Accuracy()
        self.p = torchmetrics.Precision()
        self.r = torchmetrics.Recall()
        self.f1 = torchmetrics.F1Score()
        self.metrics = {
            'accuracy': self.a,
            'precision': self.p,
            'recall': self.r,
            'f1': self.f1
        }
        self.loss = nn.BCELoss(weight=torch.FloatTensor(config['class_weights']))

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, *args, **kwargs):
        return self.base_model(**batch)

    def training_step(self, batch):
        output = self(batch)
        y_hat_proba = self.sigmoid(output.logits[:, 0])
        loss = self.loss(y_hat_proba, batch['labels'].float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        y_hat_binary = output.logits.argmax(-1)
        y_hat_proba = self.sigmoid(output.logits[:, 0])

        return {'loss': self.loss(y_hat_proba, batch['labels'].float())} | {
            key: value(y_hat_binary, batch['labels']) for key, value in self.metrics.items()
        }

    def validation_epoch_end(self, outputs):
        avg_outputs = {key: torch.stack([x[key] for x in outputs]).mean() for key in outputs[0].keys()}
        for k, v in avg_outputs.items():
            self.log(f"val_{k}", v)

    def configure_optimizers(self):
        return AdamW(self.parameters())
