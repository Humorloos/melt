import json
import logging
import numpy as np
from math import ceil
from pathlib import Path
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining, HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from sklearn.utils.class_weight import compute_class_weight

from FlushingReporter import FlushingReporter
from kbert.constants import NUM_SAMPLES, MAX_VAL_SET_SIZE, DEBUG, MATCHING_ML_DIR, DEFAULT_CONFIG, MAX_EPOCHS, RESUME, \
    RUN_NAME, SEARCH_RESTORE_DIR_NAME
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
    training_arguments.pop("save_at_end", None)  # delete if existent
    weight_of_positive_class = training_arguments.get("weight_of_positive_class", 0.5)
    training_arguments.pop("weight_of_positive_class", None)  # delete if existent
    is_tm_modification_enabled = request_headers.get('tm', 'false').lower() == 'true'
    do_tune = request_headers.get('tune', 'false').lower() == 'true'
    batch_size = training_arguments["per_device_train_batch_size"]
    training_file = request_headers["training-file"]
    log.info(f'Batch size: {batch_size}')

    data_left, data_right, labels = transformers_read_file(training_file, True)
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

    # determine val check interval (spend 10 times more time on training than on validation)
    complete_data_size = len(labels)
    val_set_size = min(MAX_VAL_SET_SIZE, complete_data_size // 10)  # size of validation set
    train_set_size = complete_data_size - val_set_size  # size of training set
    val_check_interval = 1 / (ceil(1 / (10 * val_set_size / train_set_size)))

    config = {
        'base_model': request_headers["model-name"],
        'num_labels': 2,
        'tm': is_tm_modification_enabled,
        'tm_attention': is_tm_modification_enabled and request_headers.get('tm-attention', 'false').lower() == 'true',
        'max_input_length': int(request_headers['max-length']),
        'data_dir': training_file,
        'save_dir': request_headers["resulting-model-location"],
        'index_file_path': training_arguments.get('index_file', get_index_file_path(training_file)),
        'gpus': request_headers.get("cuda-visible-devices", None),
        'val_check_interval': val_check_interval,
    }
    if do_tune:
        log.info('Setting up for HP optimization')
        # import ray
        # ray.init(
        #     # for debugging:
        #     local_mode=DEBUG,
        #     include_dashboard=False,
        #     num_cpus={True: 1, False: 12}[DEBUG],
        #     num_gpus=1,
        #     log_to_driver=True,
        # )

        search_config = {
            'batch_size_train': tune.qloguniform(2, batch_size, q=1),
            'lr': tune.loguniform(1e-7, 1e-2),
            'weight_decay': tune.loguniform(1e-7, 1e-1),
            'dropout_prob': tune.uniform(0.1, 0.5),
            'positive_class_weight': tune.uniform(0.1, 0.9)
        }

        # Reporter for reporting progress in command line
        reporter = FlushingReporter(
            parameter_columns=["batch_size_train", "lr", "weight_decay", "dropout_prob", "positive_class_weight"],
            metric_columns=["loss", "p", "r", "f1", "training_iteration"]
        )
        ray_local_dir = Path(tmp_dir) / "ray_local_dir"

        # PBT
        # scheduler = PopulationBasedTraining(
        #     time_attr="training_iteration",
        #     perturbation_interval=1,
        #     hyperparam_mutations=search_config,
        # )

        # BOHB
        # BOHB search algorithm for finding new hyperparameter configurations
        search_alg = TuneBOHB()
        if SEARCH_RESTORE_DIR_NAME is not None:
            search_alg.restore_from_dir(ray_local_dir / SEARCH_RESTORE_DIR_NAME)
        search_alg.set_search_properties(metric='f1', mode='max', config=search_config)

        # BOHB scheduler for scheduling and discarding trials
        iterations_per_epoch = 1 / val_check_interval
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            # train for at most the number of iterations that fit into the max number of epochs
            max_t=MAX_EPOCHS * iterations_per_epoch
        )

        log.info('Starting Ray Tune run')
        return tune.run(
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
            max_failures=-1,  # default: 0, set to higher value to re-try failed trials
            resources_per_trial={
                'gpu': 1,
                'cpu': 1
            },
            search_alg=search_alg,
            progress_reporter=reporter,
            callbacks=[WandbLoggerCallback(
                project='master_thesis',
                api_key_file=MATCHING_ML_DIR / 'wandb_api_key',
                log_config=True
            )]
        )
    else:
        config.update(DEFAULT_CONFIG)

        return train_transformer(config, do_tune=do_tune)
