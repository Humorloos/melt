from ray import tune
from ray.train import Trainer

from train_transformer import train_transformer


def ray_train(config):
    trainer = Trainer(backend="torch", num_workers=1, use_gpu=True)
    trainer.start()
    results = trainer.run(tune.with_parameters(
        train_transformer,
        do_tune=True
    ), config)
    trainer.shutdown()
    return results
