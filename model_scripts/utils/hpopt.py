import json
from pathlib import Path
from lightning import pytorch as pl
import os

import ray
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer)
from ray.train.torch import TorchTrainer
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler

from utils.creator import create_loader, create_model


def train_model(config, train_dset, val_dset, test_dset, num_workers, scaler, train_targets, test_targets, weights, extra_dim):
    train_loader, val_loader, test_loader = create_loader(
        config, train_dset, val_dset, test_dset, weights)
    model = create_model(config, scaler, train_targets=train_targets, test_targets=test_targets, extra_dim=extra_dim)

    trainer = pl.Trainer(
        max_epochs=50,  # number of epochs to train for
        # below are needed for Ray and Lightning integration
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_loader, val_loader)


def hpopt(train_dset,
          val_dset,
          test_dset,
          scaler,
          conf_path: str,
          result_path: str,
          train_targets: list[str],
          test_targets: list[str],
          weight: list[float | int],
          extra_dim: int | float,
          resume_ckpt_path=None,
          name: str = "tdc",
          search_alg=HyperOptSearch(n_initial_points=1, random_state_seed=42)
          ):
    # Start tuning
    ray.init() #logging_level='debug')

    search_space = {
        "depth": tune.qrandint(lower=2, upper=4, q=1),
        "ffn_hidden_dim": tune.qrandint(lower=100, upper=1000, q=100),
        "ffn_num_layers": tune.qrandint(lower=1, upper=3, q=1),
        "message_hidden_dim": tune.qrandint(lower=100, upper=1000, q=100),
        # e.g., between 0.1 and 0.5
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-5, 1e-2),           # learning rate, log scale
        "warmup": tune.choice([1,2]),  # warmup steps/epochs
        "batch_size": tune.choice([32, 64, 128]),  # batch sizes
        "weight_decay": tune.loguniform(1e-5, 1e-2)
    }
    scheduler = FIFOScheduler()

    # Scaling config controls the resources used by Ray
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,  # change to True if you want to use GPU
    )

    # Checkpoint config controls the checkpointing behavior of Ray
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,  # number of checkpoints to keep
        # Save the checkpoint based on this metric
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",  # Save the checkpoint with the lowest metric value
    )

    hpopt_save_dir = Path.cwd().parent / "hyper_param" / name

    run_config = RunConfig(
        checkpoint_config=checkpoint_config,
        storage_path=hpopt_save_dir  # directory to save the results
    )

    ray_trainer = TorchTrainer(
        lambda config: train_model(
            config, train_dset, val_dset, test_dset, 0, scaler, train_targets, test_targets, weight, extra_dim
        ),
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        num_samples=50,  # number of trials to run
        scheduler=scheduler,
        search_alg=search_alg,
        trial_dirname_creator=lambda trial: str(
            trial.trial_id),  # shorten filepaths

    )

    if resume_ckpt_path is not None:
        tuner = tune.Tuner.restore(
            path=resume_ckpt_path,
            trainable=ray_trainer,
            resume_unfinished=True
        )
    else:
        tuner = tune.Tuner(
            ray_trainer,
            param_space={
                "train_loop_config": search_space,
            },
            tune_config=tune_config,
        )
    results = tuner.fit()

    # Save the best configuration and result table

    best_config = results.get_best_result().config
    with open(conf_path, 'w') as f:
        json.dump(best_config, f, indent=4)

    results_df = results.get_dataframe()
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results_df.to_csv(result_path)

    ray.shutdown()
