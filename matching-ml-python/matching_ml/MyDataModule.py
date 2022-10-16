import pandas as pd
import pytorch_lightning as pl
import torch
from math import floor, ceil
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding

import wandb
from MyRawDataset import MyRawDataset
from MyTokenizedDataset import MyTokenizedDataset
from WandbFile import WandbFile
from kbert.constants import MAX_VAL_SET_SIZE, MAX_EPOCH_EXAMPLES
from kbert.tokenizer.constants import RANDOM_STATE
from kbert.utils import print_time
from utils import transformers_read_file, initialize_tokenizer, transformers_get_df


class MyDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_path: str = None,
            predict_data_path: str = None,
            test_data_path: str = None,
            batch_size: int = 64,
            num_workers=1,
            tokenize=True,
            tm=False,
            base_model=None,
            max_input_length=None,
            tm_attention=False,
            index_file_path=None,
            one_epoch=False,
            **kwargs
    ):
        super().__init__()
        self.data_train = None
        self.data_val = None
        self.data_predict = None
        self.data_test = None
        if train_data_path is None:
            self.train_data_path = train_data_path
        else:
            self.train_data_path = Path(train_data_path)
        self.predict_data_path = predict_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = [num_workers, 0][num_workers == 1]
        if tokenize:
            self.tokenizer = initialize_tokenizer(
                is_tm_modification_enabled=tm,
                model_name=base_model,
                max_length=max_input_length,
                tm_attention=tm_attention,
                index_file_path=index_file_path
            )
        else:
            self.tokenizer = None
        self.one_epoch = one_epoch

    def setup(self, stage=None, epoch=None, condensation_factor=None, **kwargs):
        if stage in ['fit', 'validate'] and self.data_train is None:
            val_data_path = self.train_data_path.parent / self.train_data_path.name.replace("train", "val")
            separate_val_file = (val_data_path).exists()
            if self.tokenizer is None:
                print('load pre-tokenized dataset')
                # with open(self.train_data_path, 'rb') as train_data_in:
                #     train_data_dict = torch.load(train_data_in)
                # for k, v in train_data_dict.items():
                #     dim = len(v[0].shape)
                #     if dim == 1:
                #         concatenated_inputs = torch.cat(v)
                #     elif dim == 2:
                #         concatenated_inputs = torch.cat([pad(a, (0, max_len - a.shape[1])) for a in v])
                #     else:
                #         concatenated_inputs = torch.cat([pad(a, (0, max_len - a.shape[2], 0, max_len - a.shape[2])) for a in v])
                #     train_data_dict[k] = list(concatenated_inputs.detach().numpy())
                #
                # df = pd.DataFrame(train_data_dict)
                with print_time('loading pickled train df'):
                    df = pd.read_pickle(self.train_data_path)
            else:
                print("Prepare transformer dataset")
                df = transformers_get_df(self.train_data_path, True)
            complete_data_size = df.shape[0]  # size of training + validation set (before split)
            print(f"Transformer dataset contains {complete_data_size} examples.")
            val_set_size = min(MAX_VAL_SET_SIZE, complete_data_size // 5)  # size of validation set
            if separate_val_file:
                df_train = df
                if self.tokenizer is None:
                    with print_time('loading pickled val df'):
                        df_val = pd.read_pickle(val_data_path)
                else:
                    df_val = transformers_get_df(val_data_path, True)
                if df_val.shape[0] > val_set_size:
                    val_fraction = val_set_size / df_val.shape[0]
                    df_val = df_val.groupby(['label'], group_keys=False).apply(
                        lambda df: df.sample(
                            {
                                0: floor(val_fraction * df.shape[0]),
                                1: ceil(val_fraction * df.shape[0])
                            }[RANDOM_STATE.binomial(1, val_fraction)],
                            random_state=RANDOM_STATE
                        )
                    )
            else:
                df_train, df_val = train_test_split(
                    df,
                    test_size=val_set_size,
                    random_state=RANDOM_STATE,
                    # stratify=labels[:128]
                    stratify=df['label']
                )
            if condensation_factor is not None:
                grouped_df_train = df_train.groupby('label', group_keys=False)
                n_pos_examples = grouped_df_train.size()[1]

                def sample_or_not(d):
                    if d.name == 0:
                        return d.sample(min(d.shape[0], int(condensation_factor * n_pos_examples)),
                                        random_state=RANDOM_STATE)
                    return d

                df_train = grouped_df_train.apply(sample_or_not)
            self.data_train = df_train
            self.data_val = df_val

        elif stage == 'predict' and self.data_predict is None:
            print("Preparing data for predict stage")
            data_left, data_right, _ = transformers_read_file(self.predict_data_path, False)
            assert len(data_left) == len(data_right)
            self.data_predict = MyRawDataset(texts_left=data_left, texts_right=data_right)

        elif stage == 'test' and self.data_test is None:
            print('Preparing data for test stage')
            data_left, data_right, labels = transformers_read_file(self.test_data_path, True)
            assert len(data_left) == len(data_right) == len(labels)
            self.data_test = MyRawDataset(texts_left=data_left, texts_right=data_right, labels=labels)

    def train_dataloader(self):
        train_size = self.data_train.shape[0]
        # if epoch is not None:
        #     df_train = self.data_train.sample(frac=1, random_state=RANDOM_STATE)
        #     n_train_splits = ceil(df_train.shape[0] / MAX_EPOCH_EXAMPLES)
        #     current_split_offset = epoch % n_train_splits
        #     split_start = current_split_offset * MAX_EPOCH_EXAMPLES
        #     split_end = (current_split_offset + 1) * MAX_EPOCH_EXAMPLES
        #     df_train = df_train.iloc[split_start:split_end]
        if train_size > MAX_EPOCH_EXAMPLES:
            fraction_2_sample = MAX_EPOCH_EXAMPLES / train_size
            # sample without random state to get different sample each time
            df_train = self.data_train.groupby('label').apply(lambda d: d.sample(int(fraction_2_sample * d.shape[0])))
            # shuffle
            df_train = df_train.sample(frac=1, random_state=RANDOM_STATE)
        else:
            df_train = self.data_train

        print(f'Validating on {self.data_val.shape[0]} examples')
        print(f'Training on {df_train.shape[0]} examples')
        if self.tokenizer is None:
            train_data = MyTokenizedDataset(df_train)

        else:
            train_data = MyRawDataset(
                texts_left=df_train['text_left'].tolist(),
                texts_right=df_train['text_right'].tolist(),
                labels=df_train['label'].tolist()
            )
        if self.one_epoch:
            self.data_train = None
        data_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                                 collate_fn=self.collate, )
        if wandb.run is not None:
            return tqdm(data_loader, mininterval=5, file=WandbFile())
        return data_loader

    def val_dataloader(self):
        if self.tokenizer is None:
            val_data = MyTokenizedDataset(self.data_val)
        else:
            val_data = MyRawDataset(
                texts_left=self.data_val['text_left'].tolist(),
                texts_right=self.data_val['text_right'].tolist(),
                labels=self.data_val['label'].tolist()
            )
        return DataLoader(
            val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size, collate_fn=self.collate)

    def collate(self, batch):
        if self.tokenizer is None:
            encodings = BatchEncoding(data={
                k: torch.stack([d[k] for d in batch]) for k in batch[0].keys() if k != 'label'
            })
        else:
            encodings = self.tokenizer(
                text=[i['text_left'] for i in batch],
                text_pair=[i['text_right'] for i in batch],
                return_tensors='pt',
                padding=True,
                truncation='longest_first',
            )
        if 'label' in batch[0]:
            try:
                labels = torch.LongTensor([int(i['label']) for i in batch])
            except:
                print('')
            # # following lines are for analysis
            # slmr_left, slmr_right = get_single_line_molecule_representations([i['text_left'] for i in batch], self.tokenizer, self.tokenizer.base_tokenizer.model_max_length, [i['text_right'] for i in batch])
            # slmr_df = pd.DataFrame({'slmr_left': slmr_left, 'slmr_right': slmr_right, 'label': labels})
            # slmr_pos = slmr_df[slmr_df['label'] == 1]
            # numpy_encodings = {k: v.detach().numpy() for k, v in encodings.data.items()}
            return encodings, labels
        else:
            return encodings
