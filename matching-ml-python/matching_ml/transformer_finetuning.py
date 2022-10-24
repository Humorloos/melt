from collections import defaultdict

import json
import logging
import numpy as np
import pandas as pd
import re
import shutil
import torch.multiprocessing
from math import ceil
from pathlib import Path
from pytorch_lightning.utilities.cloud_io import load
from ray import tune
from ray.tune import ExperimentAnalysis
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.functional import pad
from tqdm import tqdm

from CustomPBT import CustomPBT
from DescStr import DescStr
from EarlyStopper import EarlyStopper
from FlushingReporter import FlushingReporter
from MyDataModule import MyDataModule
from MyRawDataset import MyRawDataset
from custom_explore import custom_explore
from kbert.constants import NUM_SAMPLES, DEBUG, DEFAULT_CONFIG, RESUME, \
    RUN_NAME, WORKERS_PER_TRIAL, GPUS_PER_TRIAL, PATIENCE, TARGET_METRIC, MAX_LENGTH, MIN_DELTA, MAX_EPOCH_EXAMPLES
from kbert.tokenizer.constants import RANDOM_STATE
from kbert.utils import get_best_trial
from train_transformer import train_transformer
from utils import transformers_init, get_index_file_path, get_timestamp, transformers_get_df

MAX_CONDENSATION_FACTOR = 100

MIN_CONDENSATION_FACTOR = 1

SPLIT_SIZE = 500

log = logging.getLogger('python_server_melt')
DF_FILE_COLS = ['text_left', 'text_right', 'label']
INDEX_COLS = ['key', 'value']


def inner_transformers_finetuning(request_headers):
    request_headers['tune'] = 'true'
    try:
        track_dir = Path(request_headers['training-file'])
        tm = request_headers['tm'].lower() == 'true'
        cross_test_case_train_file_path = join_test_case_dfs(tm, track_dir, 'train')
        join_test_case_dfs(tm, track_dir, 'val')
        request_headers['training-file'] = cross_test_case_train_file_path
        request_headers['model-name'] = Path(request_headers['model-name']) / 'checkpoint'

        best_model_path = finetune_transformer(request_headers)

        target_path = Path(request_headers['resulting-model-location'])
        target_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model_path, target_path)
        return "True"
    except Exception as e:
        import traceback

        return "ERROR " + traceback.format_exc()


def join_test_case_dfs(tm, track_dir, purpose):
    cross_test_case_df, test_case_indices = get_cross_test_case_df_and_indices(purpose, tm, track_dir)
    data_path = (Path(''), Path('normalized') / 'all_targets' / 'isMulti_true')[tm] / 'posref0.2_all_negatives'
    cross_test_case_dir = track_dir / 'crosstestcase' / data_path
    cross_test_case_dir.mkdir(exist_ok=True, parents=True)
    cross_test_case_path = cross_test_case_dir / f'{purpose}.csv'
    cross_test_case_df.to_csv(cross_test_case_path, header=False)
    if tm:
        track_index = pd.concat(test_case_indices, ignore_index=True)
        track_index = track_index[~track_index['key'].duplicated()].set_index('key').fillna('UNK')
        track_index.to_csv(cross_test_case_dir / f'index_{purpose}.csv', header=False)
    return cross_test_case_path


def get_cross_test_case_df_and_indices(purpose, tm, track_dir):
    test_case_dfs = []
    test_case_indices = []
    data_path = (Path(''), Path('normalized') / 'all_targets' / 'isMulti_true')[tm] / 'posref0.2_all_negatives'
    for test_case_dir in track_dir.iterdir():
        if test_case_dir.name != 'crosstestcase':
            train_dir = test_case_dir / data_path
            test_case_dfs.append(pd.read_csv(train_dir / f'{purpose}.csv', names=DF_FILE_COLS))
            if tm:
                test_case_indices.append(pd.read_csv(train_dir / f'index_{purpose}.csv', names=INDEX_COLS))
    cross_test_case_df = pd.concat(test_case_dfs, ignore_index=True)
    return cross_test_case_df, test_case_indices


def finetune_transformer(request_headers):
    torch.multiprocessing.set_sharing_strategy('file_system')
    transformers_init(request_headers)
    tmp_dir = request_headers["tmp-dir"]
    training_arguments = json.loads(request_headers["training-arguments"])
    # save_at_end = training_arguments.get("save_at_end", True)
    is_tm_modification_enabled = request_headers.get('tm', 'false').lower() == 'true'
    do_tune = request_headers.get('tune', 'false').lower() == 'true'
    batch_size = training_arguments["per_device_train_batch_size"]
    training_file = request_headers["training-file"]
    log.info(f'Max batch size: {batch_size}')
    model_name = request_headers["model-name"]
    max_length = int(request_headers['max-length'])
    tm_attention = is_tm_modification_enabled and request_headers.get('tm-attention', 'false').lower() == 'true'
    index_file_path = training_arguments.get('index_file', get_index_file_path(training_file))

    # Generate pre-tokenized file
    training_file_path = Path(training_file)
    pre_tokenized_dir = training_file_path.parent / str(max_length)
    pre_tokenized_dir.mkdir(parents=True, exist_ok=True)

    encodings_df = pre_tokenize_file(
        index_file_path,
        is_tm_modification_enabled,
        max_length,
        model_name,
        pre_tokenized_dir=pre_tokenized_dir,
        tm_attention=tm_attention,
        training_file=training_file_path
    )
    labels = encodings_df['label']
    patience = min(PATIENCE, int(encodings_df.shape[0] / MAX_EPOCH_EXAMPLES))

    val_file_path = training_file_path.parent / training_file_path.name.replace('train', 'val')
    if val_file_path.exists():
        index_file_path_ = Path(index_file_path)
        pre_tokenize_file(
            index_file_path_.parent / index_file_path_.name.replace('train', 'val'),
            is_tm_modification_enabled,
            max_length,
            model_name,
            pre_tokenized_dir,
            tm_attention,
            val_file_path
        )

    # determine val check interval (spend 10 times more time on training than on validation)
    complete_data_size = len(labels)
    # val_set_size = min(MAX_VAL_SET_SIZE, complete_data_size // 10)  # size of validation set
    # train_set_size = complete_data_size - val_set_size  # size of training set
    # val_check_interval = 1 / (ceil(1 / (10 * val_set_size / train_set_size)))
    val_check_interval = 1
    # validations_per_epoch = 1 / val_check_interval
    validations_per_epoch = 1
    num_workers = {True: 1, False: WORKERS_PER_TRIAL}[DEBUG]

    config = {
        'base_model': model_name,
        'num_labels': 2,
        'tm': is_tm_modification_enabled,
        'tm_attention': tm_attention,
        'max_input_length': max_length,
        'data_dir': training_file,
        'save_dir': request_headers["resulting-model-location"],
        'index_file_path': index_file_path,
        'gpus': request_headers.get("cuda-visible-devices", None),
        'val_check_interval': val_check_interval,
    }

    if do_tune:
        log.info('Setting up for HP optimization')
        import ray
        ray.init(
            # for debugging:
            local_mode=DEBUG,
            include_dashboard=False,
        )

        #
        n_pos_labels = sum(1 for label in labels if label == 1)
        positive_fraction = n_pos_labels / complete_data_size
        positive_class_weight = 1 - positive_fraction
        positive_class_weight_upper_bound = min(0.99, positive_class_weight + 0.125)
        positive_class_weight_lower_bound = min(0.75, positive_class_weight - 0.125)

        config['pos_frac'] = positive_fraction

        search_config = {
            'batch_size': tune.quniform(2, batch_size, q=1),
            'lr': tune.loguniform(1e-7, 1e-5),
            'weight_decay': tune.loguniform(1e-7, 1e-1),
            'dropout': tune.uniform(0.1, 0.5),
            'pos_weight': tune.uniform(positive_class_weight_lower_bound, positive_class_weight_upper_bound),
            'condense': tune.loguniform(MIN_CONDENSATION_FACTOR, MAX_CONDENSATION_FACTOR),
        }
        config = config | search_config

        experiment_name = {True: RUN_NAME, False: get_timestamp()}[RESUME]
        # Reporter for reporting progress in command line
        reporter = FlushingReporter(
            parameter_columns=["batch_size", "lr", "weight_decay", "dropout", "pos_weight", "condense"],
            metric_columns=["loss", "p", "r", "f1", "f2", 'auc', 'bin_f1', "training_iteration"],
            max_report_frequency=20,
        )
        ray_local_dir = Path(tmp_dir) / "ray_local_dir"

        # PBT
        # scheduler = PopulationBasedTraining(
        scheduler = CustomPBT(
            experiment_name=experiment_name,
            time_attr="training_iteration",
            perturbation_interval=validations_per_epoch,
            hyperparam_mutations=search_config,
            custom_explore_fn=custom_explore,
            quantile_fraction=0.5,
            # burn_in_period=10,
        )
        search_alg = None
        # PB2
        # from ray.tune.schedulers.pb2 import PB2
        # hyperparam_bounds = {
        #     'batch_size': [2, batch_size],
        #     'lr': [1e-7, 1e-5],
        #     'weight_decay': [1e-7, 1e-1],
        #     'dropout': [0.1, 0.5],
        #     'pos_weight': [positive_class_weight_lower_bound, positive_class_weight_upper_bound],
        #     'condense': [1, 100],
        # }
        # scheduler = PB2(
        #     time_attr="training_iteration",
        #     perturbation_interval=validations_per_epoch,
        #     hyperparam_bounds=hyperparam_bounds,
        # )
        # search_alg = None

        # BOHB
        # # BOHB search algorithm for finding new hyperparameter configurations
        # search_alg = TuneBOHB()
        # if SEARCH_RESTORE_DIR_NAME is not None:
        #     search_alg.restore_from_dir(ray_local_dir / SEARCH_RESTORE_DIR_NAME)
        # search_alg.set_search_properties(metric='f1', mode='max', config=search_config)
        #
        # # BOHB scheduler for scheduling and discarding trials
        # iterations_per_epoch = 1 / val_check_interval
        # scheduler = HyperBandForBOHB(
        #     time_attr="training_iteration",
        #     # train for at most the number of iterations that fit into the max number of epochs
        #     max_t=MAX_EPOCHS * iterations_per_epoch
        # )

        log.info('Starting Ray Tune run')

        def get_trial_name(trial):
            """Function for generating trial names"""
            return f"{experiment_name}_{trial.trial_id}"

        early_stopper = EarlyStopper(
            patience=patience,
            min_delta=MIN_DELTA,
            experiment_name=experiment_name,
            target_metric=TARGET_METRIC,
            ray_local_dir=ray_local_dir,
        )
        tune.run(
            # ray_train,
            tune.with_parameters(
                train_transformer,
                do_tune=True
            ),
            metric=TARGET_METRIC,
            mode="max",
            config=config,
            num_samples={True: 2, False: NUM_SAMPLES}[DEBUG],
            scheduler=scheduler,
            # local_dir=str(Path(tmpdirname) / "ray_local_dir"),
            local_dir=str(ray_local_dir),
            trial_name_creator=get_trial_name,
            trial_dirname_creator=get_trial_name,
            name=experiment_name,
            resume=RESUME,
            max_failures=-1,  # default: 0, set to higher value to re-try failed trials
            resources_per_trial={
                'gpu': GPUS_PER_TRIAL,
                'cpu': GPUS_PER_TRIAL * num_workers,
            },
            search_alg=search_alg,
            progress_reporter=reporter,
            stop=early_stopper,
            log_to_file=True,
        )
        ray.shutdown()
        return str(early_stopper.best_checkpoint.absolute())
    else:
        config.update(DEFAULT_CONFIG)

        weight_of_positive_class = training_arguments.get("weight_of_positive_class", 0.5)
        class_weights = np.array([0.5, 0.5])  # only one label available -> default to [0.5, 0.5]
        if weight_of_positive_class >= 0.0:
            # calculate class weights
            if weight_of_positive_class > 1.0:
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)

            else:
                class_weights = [1.0 - weight_of_positive_class, weight_of_positive_class]
        log.info("Using class weights: " + str(class_weights))
        config['pos_weight'] = class_weights[1]

        config['batch_size'] = batch_size

        return train_transformer(config, do_tune=do_tune)
        # return train_transformer(config, checkpoint_dir=CHECKPOINT_PATH, do_tune=do_tune)


def pre_tokenize_file(index_file_path, is_tm_modification_enabled, max_length, model_name, pre_tokenized_dir: Path,
                      tm_attention, training_file):
    # todo: split training file into equal-size splits such that each split contains
    # min_c: min condensation factor
    # max_c: max condensation factor
    # N: train epoch examples
    # S: train set size
    # pn: fraction of negatives in train set
    # nn: number of negatives in train set
    # ns: number of splits
    # n_pos = 1/(1+min_c)*N
    # n_neg = min(max_c * n_pos, nn/ns)
    # save only relevant validation examples to file
    batch_test = 16
    model_path = Path(model_name)
    if model_path.exists():
        base_model_name = load(model_path)['hyper_parameters']['config']['base_model']
    else:
        base_model_name = model_name
    df = transformers_get_df(training_file, True)
    purpose = training_file.stem
    purpose_dir = pre_tokenized_dir / purpose
    label_dict_list = []
    for label in [0, 1]:
        label_df = df[df['label'] == label].sample(frac=1, random_state=RANDOM_STATE)
        n_splits = ceil(label_df.shape[0] / SPLIT_SIZE)
        split_sizes = [len(s) for s in np.array_split(np.ones(label_df.shape[0]), n_splits)]
        label_dir = purpose_dir / f'label_{label}'
        label_dir.mkdir(exist_ok=True, parents=True)
        split_ids = {int(re.search(r'\d+', n.name)[0]) for n in label_dir.iterdir()}
        n_tokenized_splits = len(split_ids)
        label_dict_list.append({
            'df': label_df,
            'n_tokenized_splits': n_tokenized_splits,
            'done': n_splits == n_tokenized_splits,
            'dir': label_dir,
            'split_sizes': split_sizes,
        })

    if all(d['done'] for d in label_dict_list):
        print('All examples have already been tokenized, skipping pre-tokenization')
        return df

    data_module = MyDataModule(test_data_path=training_file, batch_size=batch_test,
                               # num_workers=int(os.cpu_count() / 2 ** 4),
                               num_workers=12,
                               tm=is_tm_modification_enabled,
                               base_model=base_model_name, max_input_length=max_length, tm_attention=tm_attention,
                               index_file_path=index_file_path)
    for label, label_dict in enumerate(label_dict_list):  # todo: undo reverse
        if label_dict['done']:
            print(f'All examples with label {label} have already been tokenized')
        n_tokenized_splits = label_dict['n_tokenized_splits']
        print(f'Pre-tokenizing examples with label {label} starting with split {n_tokenized_splits}')
        n_cached_examples = sum(split_sizes[:n_tokenized_splits])
        label_df = label_dict['df'].iloc[n_cached_examples:]
        data_module.data_test = MyRawDataset(
            label_df['text_left'].tolist(),
            label_df['text_right'].tolist(),
            label_df['label'].tolist()
        )
        desc = DescStr()
        df_dict = defaultdict(lambda: [])
        dataloader_iter = iter(data_module.test_dataloader())
        split_sizes = label_dict['split_sizes']
        for i, split_size in enumerate(pbar := tqdm(split_sizes[n_tokenized_splits:]),
                                       n_tokenized_splits):
            encodings_dict = defaultdict(lambda: [])
            n_batches_this_iter = ceil((split_size - len(df_dict['input_ids'])) / batch_test)
            for _ in tqdm(range(n_batches_this_iter), file=desc):
                pbar.set_description(desc.read())
                encoding, labels = next(dataloader_iter)
                encodings_dict['label'].append(labels.bool())
                for k, v in encoding.data.items():
                    if k in {'token_type_ids', 'attention_mask'}:
                        encodings_dict[k].append(v.bool())
                    elif k == 'position_ids' and MAX_LENGTH <= 2 ** 8:
                        encodings_dict[k].append(v.byte())
                    else:
                        encodings_dict[k].append(v.short())
            max_len = max(a.shape[1] for a in encodings_dict['input_ids'])
            for k, v in encodings_dict.items():
                dim = len(v[0].shape)
                if dim == 1:
                    concatenated_inputs = torch.cat(v)
                elif dim == 2:
                    concatenated_inputs = torch.cat([pad(a, (0, max_len - a.shape[1])) for a in v])
                else:
                    concatenated_inputs = torch.cat(
                        [pad(a, (0, max_len - a.shape[2], 0, max_len - a.shape[2])) for a in v])
                df_dict[k].extend(list(concatenated_inputs.detach().numpy()))
            encodings_df = pd.DataFrame({k: v[:split_size] for k, v in df_dict.items()})
            encodings_df.to_pickle(label_dict['dir'] / f'split_{i}.pickle')
            df_dict = {k: v[split_size:] for k, v in df_dict.items()}
    return df
