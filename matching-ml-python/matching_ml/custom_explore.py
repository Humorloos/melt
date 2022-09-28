from kbert.constants import BATCH_SIZE


def custom_explore(config):
    config['batch_size'] = min(BATCH_SIZE, max(2, config['batch_size']))
    config['dropout'] = min(0.5, max(0.1, config['dropout']))
    config['pos_weight'] = min(0.9999, max(0.0001, config['pos_weight']))
    return config
