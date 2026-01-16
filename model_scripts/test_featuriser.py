from featuriser.maplight import get_maplight
from featuriser.recursive_mordred import get_mordred
from featuriser.padel import get_padel_df
from featuriser.uma_features import UMAFeaturiser
import numpy as np
from chemprop import data
from chemprop.utils import make_mol
from tdc.benchmark_group import admet_group
import argparse

ap = argparse.ArgumentParser(description="Script to test featuriser output")
ap.add_argument(
    "-ft", "--feat", default="uma"
)
ap.add_argument(
    "--count",
    action="store_true"
)
args = ap.parse_args()

count = args.count
if args.feat == "uma":
    desc = UMAFeaturiser()

# Check if any element is not exactly 0 or 1
def not_binary(fp):
    if np.any((fp != 0) & (fp != 1)):
        print("Values that are not 0 or 1:", fp[(fp != 0) & (fp != 1)])
        print("Number of non-binary features:", len(fp[(fp != 0) & (fp != 1)]))
        return True
    else:
        return False

# mol = make_mol("C1CCCCC1")
# print("Start processing")
# fp = desc(mol)
# print(fp)
# print(fp.shape)
# print("Any NaN in FP:", np.isnan(fp).any())
# print("Any element that is not binary: ", not_binary(fp))

group = admet_group(path='data/')
benchmark = group.get('Caco2_Wang')
train_val = benchmark['train_val']

smis = train_val.loc[:, "Drug"].values
ys = train_val.loc[:, "Y"].values

mols = [make_mol(smi) for smi in smis]
# feeder = get_mordred(mols)
# x_d = desc(mols)
from tqdm import tqdm
dtp = [data.MoleculeDatapoint(mol, y, x_d=desc(mol)) for mol, y in tqdm(zip(mols, ys), total=len(mols))]

if isinstance(desc, UMAFeaturiser):
    print("Length of data points before:", len(dtp))
    for dp in dtp:
        if all(dp.x_d == np.zeros(128)):
            dtp.remove(dp)
    print("Length of data points after:", len(dtp))

print(dtp[0].x_d)
print(dtp[0].x_d.shape)
if isinstance(desc, UMAFeaturiser):
    print(f"{len(desc.failures)} molecules can't have UMA embeddings: {desc.failures}")
print("Any NaN in x_d:", np.isnan(dtp[0].x_d).any())
print("Any element that is not binary: ", not_binary(dtp[0].x_d))