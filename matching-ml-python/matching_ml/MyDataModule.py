import logging
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from MyRawDataset import MyRawDataset
from kbert.tokenizer.constants import RANDOM_STATE
from utils import transformers_read_file, initialize_tokenizer

log = logging.getLogger('python_server_melt')


class MyDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_path: str = None,
            predict_data_path: str = None,
            batch_size: int = 64,
            num_workers=1,
            tm=False,
            base_model=None,
            max_input_length=None,
            tm_attention=None,
            index_file_path=None,
            **kwargs
    ):
        super().__init__()
        self.data_train = None
        self.data_val = None
        self.data_predict = None
        self.train_data_path = train_data_path
        self.predict_data_path = predict_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = initialize_tokenizer(
            is_tm_modification_enabled=tm,
            model_name=base_model,
            max_length=max_input_length,
            tm_attention=tm_attention,
            index_file_path=index_file_path
        )

    def setup(self, stage=None, **kwargs):
        if stage in ['fit', 'validate'] and self.data_train is None:
            log.info("Prepare transformer dataset and tokenize")
            data_left, data_right, labels = transformers_read_file(self.train_data_path, True)
            assert len(data_left) == len(data_right) == len(labels)
            # data_zipped = list(zip(data_left, data_right, labels))[:128]
            data_zipped = list(zip(data_left, data_right, labels))
            complete_data_size = len(data_zipped)  # size of training + validation set (before split)
            log.info("Transformer dataset contains %s examples.", complete_data_size)  # size of validation set
            val_set_size = min(1000, complete_data_size // 10)
            data_train, data_val = train_test_split(
                data_zipped,
                test_size=val_set_size,
                random_state=RANDOM_STATE,
                # stratify=labels[:128]
                stratify=labels
            )
            train_left, train_right, train_labels = list(map(list, zip(*data_train)))
            val_left, val_right, val_labels = list(map(list, zip(*data_val)))

            # slmr_left, slmr_right = get_single_line_molecule_representations(data_left, tokenizer, None, data_right)
            # asdf = pd.DataFrame({'left': slmr_left, 'right': slmr_right, 'label': labels})
            # asdf['label'].value_counts()
            self.data_train = MyRawDataset(texts_left=train_left, texts_right=train_right, labels=train_labels)
            self.data_val = MyRawDataset(texts_left=val_left, texts_right=val_right, labels=val_labels)

        elif stage == 'predict' and self.data_predict is None:
            log.info("Prepare transformer dataset and tokenize")
            data_left, data_right, _ = transformers_read_file(self.predict_data_path, False)
            assert len(data_left) == len(data_right)
            self.data_predict = MyRawDataset(texts_left=data_left, texts_right=data_right)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )

    # def test_dataloader(self):
    #     return DataLoader(self.data_test, batch_size=self.batch_size)
    #
    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size, collate_fn=self.collate)

    def collate(self, batch):
        encodings = self.tokenizer(
            text=[i['text_left'] for i in batch],
            text_pair=[i['text_right'] for i in batch],
            return_tensors='pt',
            padding=True,
            truncation='longest_first',
        )
        if 'label' in batch[0]:
            labels = torch.tensor([i['label'] for i in batch])
            # # following lines are for analysis
            # slmr_left, slmr_right = get_single_line_molecule_representations([i['text_left'] for i in batch], self.tokenizer, self.tokenizer.base_tokenizer.model_max_length, [i['text_right'] for i in batch])
            # slmr_df = pd.DataFrame({'slmr_left': slmr_left, 'slmr_right': slmr_right, 'label': labels})
            # slmr_pos = slmr_df[slmr_df['label'] == 1]
            # numpy_encodings = {k: v.detach().numpy() for k, v in encodings.data.items()}
            return encodings, labels
        else:
            return encodings
