import numpy as np
from chemprop.utils import make_mol
from chemprop import data
from featuriser.uma_features import UMAFeaturiser
from tqdm import tqdm

def dtp_gen(train_tup: tuple, val_tup: tuple, test_tup=None, descriptor=None):
    if descriptor is not None:
        train_mols = [make_mol(smi) for smi in train_tup[0]]
        val_mols = [make_mol(smi) for smi in val_tup[0]]
        train_data = [data.MoleculeDatapoint(smi, y, x_d=descriptor(smi)) for smi, y in tqdm(zip(train_mols, train_tup[1]), total=len(train_mols), desc="Processing train data")]
        val_data = [data.MoleculeDatapoint(smi, y, x_d=descriptor(smi)) for smi, y in tqdm(zip(val_mols, val_tup[1]), total=len(val_mols), desc="Processing val data")]
        if test_tup is not None:
            test_mols = [make_mol(smi) for smi in test_tup[0]]
            test_data = [data.MoleculeDatapoint(smi, y, x_d=descriptor(smi)) for smi, y in tqdm(zip(test_mols, test_tup[1]), total=len(test_mols), desc="Processing test data")]
        else:
            test_data = None
        if isinstance(descriptor, UMAFeaturiser):
            datasets =  [train_data, val_data, test_data] if test_tup is not None else [train_data, val_data]
            for subset in datasets:
                for dtp in subset:
                    if all(dtp.x_d == np.zeros(128)):
                        subset.remove(dtp)
            print(f"{len(descriptor.failures)} molecules failed to get UMA embeddings:\n{descriptor.failures}")
            print("Succesfully removed failed molecules")
    else:
        train_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_tup[0], train_tup[1])]
        val_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(val_tup[0], val_tup[1])]
        test_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(test_tup[0], test_tup[1])] if test_tup is not None else None

    return train_data, val_data, test_data
