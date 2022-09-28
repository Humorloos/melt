import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from time import time
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from kbert.utils import get_tm_variant, f_score, get_metrics


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
        self.base_model.config.hidden_dropout_prob = config['dropout']

        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        num_labels = config['num_labels']
        self.p = torchmetrics.Precision(num_classes=num_labels, average=None)
        self.r = torchmetrics.Recall(num_classes=num_labels, average=None)
        self.f1 = torchmetrics.F1Score(num_classes=num_labels, average=None)
        self.f2 = lambda pred, target: f_score(pred, target, beta=2)  # for f1, set beta to 1
        self.metrics = {
            'precision': self.p,
            'recall': self.r,
            'f1': self.f1,
            'f2': self.f2
        }
        self.loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0 - config['pos_weight'], config['pos_weight']])
        )

        self.softmax = nn.Softmax(dim=0)
        self.step_start = None
        self.step_end = None

    def forward(self, batch, *args, **kwargs):
        return self.base_model(**batch)

    def training_step(self, batch):
        self.step_start = time()
        if self.step_end is not None:
            self.log('time_between', self.step_start - self.step_end)
        encodings, y = batch
        output = self(encodings)
        loss = self.loss(output.logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def training_step_end(self, step_output):
        self.step_end = time()
        self.log('time', self.step_end - self.step_start, on_step=True, logger=True)

    def validation_step(self, batch, batch_idx):
        encodings, y = batch
        output = self(encodings)
        y_hat_binary = output.logits.argmax(-1)
        y_hat = self.softmax(output.logits)[:, 1]
        return {
            'loss': self.loss(output.logits, y),
            'pred': y_hat,
            'bin_pred': y_hat_binary,
            'target': y
        }

    def test_step(self, batch, batch_idx):
        encodings, y = batch
        output = self(encodings)
        y_hat_binary = output.logits.argmax(-1)
        y_hat = self.softmax(output.logits)[:, 1]
        return {'loss': self.loss(output.logits, y),
                'pred': y_hat,
                'bin_pred': y_hat_binary,
                'target': y}

    def validation_epoch_end(self, outputs):
        if len(outputs) > 0:
            target = torch.cat([x['target'] for x in outputs]).float()
            metrics = {'val_loss': torch.stack([x['loss'] for x in outputs]).mean()} | get_metrics(
                torch.cat([x['pred'] for x in outputs]), target, 'val', include_auc=True) | get_metrics(
                torch.cat([x['bin_pred'] for x in outputs]).float(), target, 'val_bin')
            self.log_dict(
                metrics,
                logger=False
            )
            wandb.log(metrics)
        else:
            self.log('val_f1', 0)

    def test_epoch_end(self, outputs) -> None:
        target = torch.cat([x['target'] for x in outputs]).float()
        self.log_dict(
            {'val_loss': torch.stack([x['loss'] for x in outputs]).mean()} |
            get_metrics(
                torch.cat([x['pred'] for x in outputs]),
                target,
                'test',
                include_auc=True
            ) |
            get_metrics(
                torch.cat([x['bin_pred'] for x in outputs]).float(),
                target,
                'test_bin'
            )
        )

    def configure_optimizers(self):
        return AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
