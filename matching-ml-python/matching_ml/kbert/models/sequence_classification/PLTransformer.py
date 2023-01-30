"""
PyTorch lightning module wrapper for both unmodified and modified cross-encoder
"""
import pytorch_lightning as pl
import torch
from time import time
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from kbert.utils import get_tm_variant, get_metrics


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

        self.loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0 - config['pos_weight'], config['pos_weight']])
        )

        self.softmax = nn.Softmax(dim=1)
        self.step_start = None
        self.step_end = None
        self.epoch_start = None
        self.validation_start = None

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
        # return loss
        y_hat_binary = output.logits.argmax(-1)
        softmax_scores = self.softmax(output.logits)
        y_hat = softmax_scores[:, 1]
        return {
            'loss': loss,
            'pred': y_hat,
            'bin_pred': y_hat_binary,
            'target': y
        }

    def training_step_end(self, step_output):
        self.step_end = time()
        self.log('time', self.step_end - self.step_start, on_step=True, logger=True)

    def on_train_epoch_start(self) -> None:
        self.epoch_start = time()

    def training_epoch_end(self, outputs) -> None:
        target = torch.cat([x['target'] for x in outputs]).float()
        metrics = {'train_epoch_loss': torch.stack([x['loss'] for x in outputs]).mean()} | get_metrics(
            torch.cat([x['pred'] for x in outputs]), target, 'train', include_auc=True) | get_metrics(
            torch.cat([x['bin_pred'] for x in outputs]).float(), target, 'train_bin')
        self.log_dict(metrics)

    def on_validation_epoch_start(self) -> None:
        if self.epoch_start is not None:
            self.log('epoch_time', time() - self.epoch_start)
            self.epoch_start = None
        self.validation_start = time()

    def validation_step(self, batch, batch_idx):
        encodings, y = batch
        output = self(encodings)
        y_hat_binary = output.logits.argmax(-1)
        softmax_scores = self.softmax(output.logits)
        y_hat = softmax_scores[:, 1]
        return {
            'loss': self.loss(output.logits, y),
            'pred': y_hat,
            'bin_pred': y_hat_binary,
            'target': y
        }

    def validation_epoch_end(self, outputs):
        if not self.trainer.sanity_checking:
            self.log('validation_time', time() - self.validation_start)
            self.validation_start = None
            if len(outputs) > 0:
                target = torch.cat([x['target'] for x in outputs]).float()
                metrics = {'val_loss': torch.stack([x['loss'] for x in outputs]).mean()} | get_metrics(
                    torch.cat([x['pred'] for x in outputs]), target, 'val', include_auc=True) | get_metrics(
                    torch.cat([x['bin_pred'] for x in outputs]).float(), target, 'val_bin')
                self.log_dict(metrics)
            else:
                self.log('val_f1', 0)

    def test_step(self, batch, batch_idx):
        encodings, y = batch
        output = self(encodings)
        y_hat_binary = output.logits.argmax(-1)
        y_hat = self.softmax(output.logits)[:, 1]
        return {'loss': self.loss(output.logits, y),
                'pred': y_hat,
                'bin_pred': y_hat_binary,
                'target': y}

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
