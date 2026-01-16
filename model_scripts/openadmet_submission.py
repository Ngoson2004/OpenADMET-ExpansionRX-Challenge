import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from lightning import pytorch as pl
from chemprop import data, models
from chemprop.utils import make_mol
from chemprop.featurizers.molecule import MorganBinaryFeaturizer
from featuriser.maplight import get_maplight
import torch
import os
import argparse

ap = argparse.ArgumentParser(description="Script to submit result to ExpansionRX-OpenADMET competition")
ap.add_argument(
    "-n", "--name",
    default="lenselink50_Morgan_fromPolaris"
)
ap.add_argument(
    "-m","--mode",
    default="pick one",
    choices=["ensemble models", "pick one"]
)
ap.add_argument(
    "-mod", "--model",
    default="last.ckpt"
)
ap.add_argument(
    "-b", "--batch-size",
    type=int,
    default=128
)
ap.add_argument(
    "--val-maes",
    nargs="+",
    type=float,
    default=[]
)
ap.add_argument(
    "-uw", "--use-weight",
    action="store_true"
)
ap.add_argument(
    "-agg", "--aggregate",
    choices=["average", "median"],
    default="average"
)
ap.add_argument(
    "-d", "--descriptor",
    default=None,
    choices=["morgan", "maplight"]
)
ap.add_argument(
    "-o", "--output",
    default="OpenADMET_ExpansionRX_submission.csv"
)
args = ap.parse_args()

test_df = pd.read_csv("../data/open_admet/openadmet_test.csv")
test_smiles = test_df["SMILES"].values

test_mols = [make_mol(smiles) for smiles in test_smiles]

if args.descriptor == "morgan":
    descriptor = MorganBinaryFeaturizer()
elif args.descriptor == "maplight":
    descriptor = get_maplight
else:
    descriptor = None

if descriptor is None:
    test_dset = data.MoleculeDataset([data.MoleculeDatapoint(mol) for mol in test_mols])
else:
    test_dset = data.MoleculeDataset([data.MoleculeDatapoint(mol, x_d=descriptor(mol)) for mol in tqdm(test_mols, total=len(test_mols), desc="Creating test dataset")])

test_loader = data.build_dataloader(test_dset, batch_size=args.batch_size, shuffle=False)

ckpt_files = [f for f in os.listdir(f"../chemprop_openadmet_ckpt/{args.name}") if f.endswith('.ckpt') and "last" in f]
print(ckpt_files)

if args.mode == "pick one":
    assert not isinstance(args.name, list), "picking one model only allows one test"
    model = models.MPNN.load_from_checkpoint(f"../chemprop_openadmet_ckpt/{args.name}/{args.model}", strict=False)
    trainer = pl.Trainer(logger=False, devices=1)
    preds = trainer.predict(model, test_loader)
    preds = torch.concat(preds)[:, :9, 0]
    print("Predictions before unlogging", preds[:5, :])
    preds[:, 1:8] = torch.pow(10, preds[:, 1:8])
    preds[:, 8] = (100*torch.pow(10, preds[:, 8])) / (torch.pow(10, preds[:,8])+1)
    print("Predictions after unlogging", preds[:5, :])
    predictions = preds.cpu().numpy()
else:
    predictions = []
    for f in ckpt_files:
            model = models.MPNN.load_from_checkpoint(f"../chemprop_openadmet_ckpt/{args.name}/{f}", strict=False)
            trainer = pl.Trainer(logger=False, devices=1)
            preds = trainer.predict(model, test_loader)
            preds = torch.concat(preds)[:, :9, 0]
            preds[:, 1:8] = torch.pow(10, preds[:, 1:8])
            preds[:, 8] = (100*torch.pow(10, preds[:, 8])) / (torch.pow(10, preds[:,8])+1)
            preds = preds.cpu().numpy()
            predictions.append(preds)

weights = [1/mae for mae in args.val_maes] if args.use_weight else None
if args.mode == "ensemble models":
    submission = np.average(predictions, axis=0, weights=weights) if args.aggregate == "average" else np.median(predictions, axis=0)
else:
    submission = predictions

print(submission.shape)
cols = pd.read_csv("../data/open_admet/openadmet_train.csv").columns[2:]
reorder_col = {
    0: 0,
    1: 4,
    2: 3,
    3: 6,
    4: 2,
    5: 1,
    6: 7,
    7: 5,
    8: 8
}
for i in range(9):
    test_df[cols[i]] = submission[:, reorder_col[i]]

test_df.to_csv(f"../submission/{args.output}", index=False)
