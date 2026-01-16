from chemprop import data
import pandas as pd
import numpy as np
from rdkit import Chem
from tdc.benchmark_group import admet_group
from utils.datapoint_generator import dtp_gen


def preprocess_data(train_df, 
                valid_df, 
                test_df=None, 
                disable_testset=False, 
                train_targets=None, 
                test_targets="Y", 
                smiles_col="smiles", 
                n1=None, 
                n2=None,
                primary_datasource="TDC", 
                descriptor=None):
    if (len(train_targets) > len(test_targets)):
        spare_col = [task for task in train_targets if task != smiles_col and task not in test_targets]
        valid_df[spare_col] = pd.DataFrame(
            [[None] * len(spare_col)], columns=spare_col)
    else:
        pass

    if test_df is None and not disable_testset: 
        group = admet_group(path='data/')
        benchmark = group.get('Caco2_Wang')
        test_df = benchmark['test']
        test_df = test_df.rename(columns={"Drug": f"{smiles_col}"})

    train_smis = train_df.loc[:, smiles_col].values
    train_ys = train_df.loc[:, train_targets].values

    val_smis = valid_df.loc[:, smiles_col].values
    val_ys = valid_df.loc[:, train_targets].values

    if not disable_testset:
        test_smis = test_df.loc[:, smiles_col].values
        test_ys = test_df.loc[:, test_targets].values
        test_tup = (test_smis, test_ys)
    else:
        test_tup = None
 
    train_data, val_data, test_data = dtp_gen((train_smis, train_ys), (val_smis, val_ys), test_tup, descriptor)

    # featuriser  = featurizers.CuikmolmakerMolGraphFeaturizer() 
    train_dset = data.MoleculeDataset(train_data)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data)
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_data) if not disable_testset else None

    extra_dim = train_dset.d_xd

    valid_smi = [Chem.MolToSmiles(dp.mol) for dp in train_data]
    weights = [
            1/n1 if source == primary_datasource else 1/n2 
            for source in train_df.loc[[smi in valid_smi for smi in train_df[f"{smiles_col}"]]]["source"]
        ] if (n1 is not None and n2 is not None) else None

    return train_dset, val_dset, test_dset, scaler, extra_dim, weights