import json
import logging
import numpy as np
from math import ceil
from pathlib import Path
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from sklearn.utils.class_weight import compute_class_weight

from FlushingReporter import FlushingReporter
from kbert.constants import NUM_SAMPLES, MAX_VAL_SET_SIZE, DEBUG, MATCHING_ML_DIR, DEFAULT_CONFIG, RESUME, \
    RUN_NAME, WORKERS_PER_TRIAL, GPU, CHECKPOINT_PATH
from train_transformer import train_transformer
from utils import transformers_init, transformers_read_file, get_index_file_path, get_timestamp, \
    get_trial_name

log = logging.getLogger('python_server_melt')


def inner_transformers_finetuning(request_headers):
    try:
        finetune_transformer(request_headers)
        return "True"
    except Exception as e:
        import traceback

        return "ERROR " + traceback.format_exc()


def finetune_transformer(request_headers):
    transformers_init(request_headers)
    tmp_dir = request_headers["tmp-dir"]
    training_arguments = json.loads(request_headers["training-arguments"])
    # save_at_end = training_arguments.get("save_at_end", True)
    is_tm_modification_enabled = request_headers.get('tm', 'false').lower() == 'true'
    do_tune = request_headers.get('tune', 'false').lower() == 'true'
    batch_size = training_arguments["per_device_train_batch_size"]
    training_file = request_headers["training-file"]
    log.info(f'Batch size: {batch_size}')

    data_left, data_right, labels = transformers_read_file(training_file, True)

    # determine val check interval (spend 10 times more time on training than on validation)
    complete_data_size = len(labels)
    val_set_size = min(MAX_VAL_SET_SIZE, complete_data_size // 10)  # size of validation set
    train_set_size = complete_data_size - val_set_size  # size of training set
    val_check_interval = 1 / (ceil(1 / (10 * val_set_size / train_set_size)))
    model_name = request_headers["model-name"]
    max_length = int(request_headers['max-length'])
    tm_attention = is_tm_modification_enabled and request_headers.get('tm-attention', 'false').lower() == 'true'
    index_file_path = training_arguments.get('index_file', get_index_file_path(training_file))
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
        num_gpus = len(GPU)
        ray.init(
            # for debugging:
            local_mode=DEBUG,
            include_dashboard=False,
        )

        #
        n_pos_labels = sum(1 for label in labels if label == 1)
        positive_class_weight = 1 - n_pos_labels / complete_data_size
        positive_class_weight_upper_bound = min(0.99, positive_class_weight + 0.125)
        positive_class_weight_lower_bound = min(0.75, positive_class_weight - 0.125)

        search_config = {
            'batch_size_train': tune.quniform(2, batch_size, q=1),
            'lr': tune.loguniform(1e-7, 1e-5),
            'weight_decay': tune.loguniform(1e-7, 1e-1),
            'dropout_prob': tune.uniform(0.1, 0.5),
            'positive_class_weight': tune.uniform(positive_class_weight_lower_bound, positive_class_weight_upper_bound)
        }

        # Reporter for reporting progress in command line
        reporter = FlushingReporter(
            parameter_columns=["batch_size_train", "lr", "weight_decay", "dropout_prob", "positive_class_weight"],
            metric_columns=["loss", "p", "r", "f1", "training_iteration"]
        )
        ray_local_dir = Path(tmp_dir) / "ray_local_dir"

        # PBT
        from ray.tune.schedulers import PopulationBasedTraining
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=1,
            hyperparam_mutations=search_config,
        )
        search_alg = None

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
        result = tune.run(
            # ray_train,
            tune.with_parameters(
                train_transformer,
                do_tune=True
            ),
            metric="f1",
            mode="max",
            config=config,
            num_samples={True: 1, False: NUM_SAMPLES}[DEBUG],
            scheduler=scheduler,
            # local_dir=str(Path(tmpdirname) / "ray_local_dir"),
            local_dir=str(ray_local_dir),
            trial_name_creator=get_trial_name,
            trial_dirname_creator=get_trial_name,
            name={True: RUN_NAME, False: get_timestamp()}[RESUME],
            resume=RESUME,
            max_failures=1,  # default: 0, set to higher value to re-try failed trials
            resources_per_trial={
                'gpu': 0.5,
                'cpu': 0.5 * num_workers,
            },
            search_alg=search_alg,
            progress_reporter=reporter,
            callbacks=[WandbLoggerCallback(
                project='master_thesis',
                api_key_file=MATCHING_ML_DIR / 'wandb_api_key',
                log_config=True
            )],
            keep_checkpoints_num=1,
            checkpoint_score_attr='f1'
        )
        ray.shutdown()
        return result
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
        config['positive_class_weight'] = class_weights[1]

        config['batch_size_train'] = batch_size

        return train_transformer(config, do_tune=do_tune)
        # return train_transformer(config, checkpoint_dir=CHECKPOINT_PATH, do_tune=do_tune)
