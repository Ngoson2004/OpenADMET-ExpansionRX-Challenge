import pandas as pd
import numpy as np
from utils.scaffold_split import scaffold_split
from sklearn.model_selection import train_test_split

def opad_split(seed=42, smiles_col="SMILES", add_novartis=False, 
                add_cdd=False,
                test_cdd=False,
                cdd_merge_key="Molecule Name",
                add_mega=False,
                only_openadmet=False, 
                use_sampler=False,
                random_split=True):
    openadmet = pd.read_csv("../data/open_admet/curated_openadmet_train_2log_1logit_wName.csv")

    if not only_openadmet:
        if add_novartis:
            helper = pd.read_csv("../data/LenseLink_data/Lenselink_Novartis.csv") #Goosen_Novartis_clean.csv")
            if add_cdd:
                cdd = pd.read_csv("../data/open_admet/CDD_Lenselink-Novartis_4tasks_400K.csv") #CDD_Lenselink-Novartis_noCanonicalise.csv")
                # cdd = cdd[["SMILES", "Log D", "pKa", "pKa (Acidic)", "pKa (Basic)"]]
                helper = pd.merge(helper, cdd, on="SMILES", how="outer")
        elif add_mega:
            helper = pd.read_csv("../data/open_admet/MEGA_Lenselink_Novartis_AZ_QM9.csv")
        else:
            helper = pd.read_csv("../data/LenseLink_data/Lenselink_Goosen_original_clean.csv")
        helper["source"] = "Helper"

    if not random_split:
        tmp_train, valid = scaffold_split(openadmet, smiles_col=smiles_col, ratios=(0.9,0.1,0.0), seed=seed)
    else:
        tmp_train, valid = train_test_split(openadmet, train_size=0.9, test_size=0.1, random_state=seed)
    
    if add_cdd:
        if not test_cdd:
            cdd = pd.read_csv("../data/open_admet/openadmet_CDD_train.csv") #pd.read_csv("../data/open_admet/CDD_ExpansioRxTrain_multiLogD.csv") 
        else:
            cdd = pd.read_csv("../data/open_admet/openadmet_CDD_test.csv")
        cdd = cdd[[f"{cdd_merge_key}", "log P", "log D", "log S", "pKa", "pKa (Acidic)", "pKa (Basic)"]]
        tmp_train = pd.merge(tmp_train, cdd, on=[f"{cdd_merge_key}"], how="outer")
        
    if not only_openadmet:
        tmp_train["source"] = "OpenADMET"
        train = pd.concat([tmp_train, helper], ignore_index=True)
        if use_sampler:
            n_helper, n_open = len(helper), len(tmp_train)
        else:
            n_helper, n_open = None, None
    else:
        train = tmp_train
        n_helper, n_open = None, None

    return train, valid, n_open, n_helper