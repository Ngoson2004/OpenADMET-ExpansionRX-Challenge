from utils.openadmet_split import opad_split
from utils.preprocess import preprocess_data
from utils.creator import create_model, create_loader

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from featuriser.maplight import get_maplight
from featuriser.uma_features import UMAFeaturiser
from chemprop.featurizers.molecule import MorganBinaryFeaturizer

import os
import argparse

ap = argparse.ArgumentParser(description="Script to start training on Chemprop")
ap.add_argument(
    "-n", "--name",
    default="openadmet",
    help="Name/identifier for this particular training/test run. Used for logging/checkpointing."
)
ap.add_argument(
    "-tn", "--test-name",
    default="openadmet_morgan",
    help="Name/identifier for this particular training/test run version. Used for logging/checkpointing."
)
ap.add_argument(
    "--conf",
    default="best_config_default.json",
    help="Path to JSON configuration file for model config. Default: best_config_default.json."
)
ap.add_argument(
    "--ckpt",
    default="orginal_openadmet_Morgan",
    help="Directory name for saving model checkpoints."
)
ap.add_argument(
    "-av", "--add-novartis",
    action="store_true",
    help="Whether to add Novartis's molecules to training set"
)
ap.add_argument(
    "-oo", "--only-openadmet",
    action="store_true",
    help="Whether to train on openadmet dataset only"
)
ap.add_argument(
    "-us", "--use-sampler",
    action="store_true",
    help="Whether to use sampler for training"
)
ap.add_argument(
    "-ft", "--featuriser",
    default=None,
    choices=["morgan","maplight", "uma"],
    help="Featurisation method to use on input molecules. Options: morgan, maplight, uma"
)
ap.add_argument(
    "-e", "--epoch",
    type=int,
    default=5
)
ap.add_argument(
    "-sc", "--scaffold",
    action="store_true",
    help="Whether to use scaffold split"
)
ap.add_argument(
    "--cdd",
    action="store_true",
    help="Wether to integrate CDD features"
)
ap.add_argument(
    "--cdd-test",
    action="store_true",
    help="Wether to integrate CDD features from molecules of blinded test set too"
)
ap.add_argument(
    "-ck", "--cdd-key",
    choices=["Molecule Name", "SMILES"],
    default="Molecule Name",
    help="Which column name to take as key to merge CDD features"
)
ap.add_argument(
    "-am", "--add-mega",
    action="store_true",
)
ap.add_argument(
    "-gc", "--grad-clip",
    type=float,
    default=None,
    help="Whether to use gradient clipping"
)

args = ap.parse_args()

predictions_list = []
conf = f'../hyper_param/{args.name}/{args.conf}'
ckpt = f'../chemprop_openadmet_ckpt/{args.ckpt}'
os.makedirs(ckpt, exist_ok=True)

if args.featuriser == "morgan":
    descriptor = MorganBinaryFeaturizer()
elif args.featuriser == "maplight":
    descriptor = get_maplight
elif args.featuriser == "uma":
    descriptor = UMAFeaturiser()
else:
    descriptor = None

predictions = []
# Perform training in different seeds
for seed in [1, 2, 3, 4, 5]:
    train, valid, n_open, n_lenselink = opad_split(seed=seed, add_novartis=args.add_novartis, add_cdd=args.cdd, test_cdd=args.cdd_test, cdd_merge_key=args.cdd_key, add_mega=args.add_mega, only_openadmet=args.only_openadmet, use_sampler=args.use_sampler, random_split=(not args.scaffold))
    train_targets = [col for col in train.columns if col not in ["SMILES", "source", "Molecule Name"]]
    test_targets = [col for col in valid.columns if col not in ["SMILES", "source", "Molecule Name"]]

    train_dset, val_dset, test_dset, scaler, extra_dim, weights = preprocess_data(train, valid, disable_testset=True, train_targets=train_targets, test_targets=test_targets, smiles_col="SMILES", n1=n_open, n2=n_lenselink, primary_datasource="OpenADMET", descriptor=descriptor)
    train_loader, val_loader, test_loader = create_loader(conf, train_dset, val_dset, test_dset, weights)
    model = create_model(conf, scaler, train_targets, test_targets, extra_dim)
    # Configure model checkpointing
    checkpointing = ModelCheckpoint( 
        ckpt,
        "best-{epoch}-{val_loss:.2f}",
        "val_loss",
        mode="min",
        save_last=True
    )

    # force auto‑increment under SLURM
    save_dir = f"./{args.name}_log"
    logger = TensorBoardLogger(save_dir=save_dir, name=f"{args.test_name}", version=seed)
    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        max_epochs=args.epoch,
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
        callbacks=[checkpointing]
    )
    trainer.fit(model, train_loader, val_loader)