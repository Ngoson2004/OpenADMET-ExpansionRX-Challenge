from utils.openadmet_split import opad_split
from utils.preprocess import preprocess_data
from utils.hpopt import hpopt

from featuriser.maplight import get_maplight
from featuriser.uma_features import UMAFeaturiser
from chemprop.featurizers.molecule import MorganBinaryFeaturizer

import argparse
from pathlib import Path

ap = argparse.ArgumentParser(description="Script to run hyper-param optimisation on Chemprop")
ap.add_argument(
    "-n", "--name",
    default="openadmet",
    help="Name of the benchmark dataset"
)
ap.add_argument(
    "--conf",
    default="best_config_Morgan_auto.json",
    help="Path to JSON configuration file for hyperparameter optimisation. Default: best_config_Maplight.json."
)
ap.add_argument(
    "--result",
    default="res_Morgan_openadmet.csv",
    help="Path to file for saving hyperparameter optimisation results. Default: result_Maplight.csv."
)
ap.add_argument(
    "-av", "--add-novartis",
    action="store_true",
    help="Whether to add Novartis's molecules to training set"
)
ap.add_argument(
    "-ft", "--featuriser",
    default="morgan",
    choices=["morgan", "maplight", "uma"],
    help="Featurisation method to use on input molecules. Options: morgan, maplight, mordred, padel, uma"
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
    "--resume",
    default=None,
    help="Optional: Path to trial directory to resume a previous Ray Tune optimisation. If not set, start a new run."
)

args = ap.parse_args()

train, valid, n_open, n_lenselink = opad_split(add_novartis=args.add_novartis, only_openadmet=args.only_openadmet, use_sampler=args.use_sampler)

train_targets = [col for col in train.columns if col not in ["SMILES", "source"]]
test_targets = [col for col in valid.columns if col not in ["SMILES", "source"]]

conf = f"../hyper_param/{args.name}/{args.conf}"
result = f"../hyper_param/result_table/{args.name}/{args.result}"
resume = str((Path.cwd().parent / "hyper_param" / args.name / args.resume).resolve()) if args.resume is not None else None

if args.featuriser == "morgan":
    descriptor = MorganBinaryFeaturizer()
elif args.featuriser == "maplight":
    descriptor = get_maplight
elif args.featuriser == "uma":
    descriptor = UMAFeaturiser()
else:
    descriptor = None

train_dset, val_dset, test_dset, scaler, extra_dim, weights = preprocess_data(train, valid, disable_testset=True, train_targets=train_targets, test_targets=test_targets, smiles_col="SMILES", n1=n_open, n2=n_lenselink, descriptor=descriptor, primary_datasource="OpenADMET")
hpopt(train_dset, val_dset, test_dset, scaler, conf, result, train_targets, test_targets, weights, extra_dim, resume, args.name)
