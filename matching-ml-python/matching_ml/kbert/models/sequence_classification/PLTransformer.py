import pytorch_lightning as pl
import torch
import torchmetrics
from copy import deepcopy
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from kbert.utils import get_tm_variant


class PLTransformer(pl.LightningModule):
    @classmethod
    def from_pretrained(cls, config, *model_args, **kwargs):
        model = AutoModelForSequenceClassification.from_pretrained(
            config['base_model'], *model_args, **kwargs
        )

        if config['tm_attention']:
            model = get_tm_variant(model)
        config['model'] = model
        return PLTransformer(config)

    def __init__(self, config=None):
        super().__init__()

        self.save_hyperparameters()

        self.base_model = config['model']
        self.base_model.config.hidden_dropout_prob = config['dropout_prob']

        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        num_labels = config['num_labels']
        self.p = torchmetrics.Precision(num_classes=num_labels, average=None)
        self.r = torchmetrics.Recall(num_classes=num_labels, average=None)
        self.f1 = torchmetrics.F1Score(num_classes=num_labels, average=None)
        self.metrics = {
            'precision': self.p,
            'recall': self.r,
            'f1': self.f1
        }
        self.loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0 - config['positive_class_weight'], config['positive_class_weight']])
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, *args, **kwargs):
        return self.base_model(**batch)

    def training_step(self, batch):
        encodings, y = batch
        output = self(encodings)
        loss = self.loss(output.logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        encodings, y = batch
        output = self(encodings)
        y_hat_binary = output.logits.argmax(-1)

        return {'loss': self.loss(output.logits, y),
                'pred': y_hat_binary,
                'target': y}

    def validation_epoch_end(self, outputs):
        if len(outputs) > 0:
            self.log(f"val_loss", torch.stack([x['loss'] for x in outputs]).mean())
            for key, value in self.metrics.items():
                self.log(f"val_{key}", value(
                    torch.cat([x['pred'] for x in outputs]),
                    torch.cat([x['target'] for x in outputs])
                )[1])
        else:
            self.log('val_f1', 0)

    def configure_optimizers(self):
        return AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
