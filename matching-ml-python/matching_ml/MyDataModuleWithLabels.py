import logging
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from kbert.constants import DEBUG
from kbert.tokenizer.constants import RANDOM_STATE
from utils import transformers_read_file, transformers_create_dataset, initialize_tokenizer

log = logging.getLogger('python_server_melt')


class MyDataModuleWithLabels(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers, tm, base_model=None, max_input_length=None, tm_attention=None,
                 index_file_path=None, **kwargs):
        super().__init__()
        self.tm = tm
        self.base_model = base_model
        self.max_input_length = max_input_length
        self.tm_attention = tm_attention
        self.index_file_path = index_file_path
        self.data_val = None
        self.data_train = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None, **kwargs):
        log.info("Prepare transformer dataset and tokenize")
        data_left, data_right, labels = transformers_read_file(self.data_dir, True)
        assert len(data_left) == len(data_right) == len(labels)
        # data_zipped = list(zip(data_left, data_right, labels))[:128]
        data_zipped = list(zip(data_left, data_right, labels))
        complete_data_size = len(data_zipped)  # size of training + validation set (before split)
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
        log.info("Transformer dataset contains %s examples.", complete_data_size)  # size of validation set
        tokenizer = initialize_tokenizer(
            is_tm_modification_enabled=self.tm,
            model_name=self.base_model,
            max_length=self.max_input_length,
            tm_attention=self.tm_attention,
            index_file_path=self.index_file_path
        )
        self.data_train = transformers_create_dataset(
            False, tokenizer, train_left, train_right, train_labels
        )
        self.data_val = transformers_create_dataset(
            False, tokenizer, val_left, val_right, val_labels
        )
        # self.data_test = MNIST(self.data_dir, train=False)
        # self.data_predict = MNIST(self.data_dir, train=False)


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.data_test, batch_size=self.batch_size)
    #
    # def predict_dataloader(self):
    #     return DataLoader(self.data_predict, batch_size=self.batch_size)
