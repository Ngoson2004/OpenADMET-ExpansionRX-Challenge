# pip install rdkit-pypi pandas numpy
from typing import Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    ratios: Tuple[float, float, float] = (0.9, 0.1, 0.0),
    seed: int = 0,
    include_chirality: bool = False,
    keep_split_col: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bemis–Murcko scaffold split -> return (train_df, val_df, test_df) with no scaffold leakage.

    Parameters
    ----------
    df : DataFrame (must contain `smiles_col`)
    smiles_col : SMILES column name
    ratios : (train, val, test) fractions; will be normalized
    seed : RNG seed for tie-breaking among equal-size scaffold buckets
    include_chirality : whether scaffold uses isomeric SMILES
    keep_split_col : if True, keep a 'split' column in the returned DataFrames

    Returns
    -------
    (train_df, val_df, test_df) : DataFrames preserving original indices
    """
    assert smiles_col in df.columns, f"Missing column: {smiles_col}"
    out = df.copy()

    # 1) Scaffold per row (stable/canonical)
    def _scaffold(smi: str) -> str:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return f"_NOSCAF_{smi}"
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaf_mol = Chem.MolFromSmiles(scaf)
        return Chem.MolToSmiles(scaf_mol, isomericSmiles=include_chirality) if scaf_mol else f"_NOSCAF_{smi}"

    out["_scaf"] = out[smiles_col].map(_scaffold)

    # 2) Buckets (scaffold -> indices), sort by size desc with seeded tie-break
    buckets = out.groupby("_scaf").indices
    rng = np.random.default_rng(seed)
    bucket_items = list(buckets.items())
    rng.shuffle(bucket_items)                              # tie-break order
    bucket_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    # 3) Targets from ratios
    ratios = np.array(ratios, float); ratios /= ratios.sum()
    n = len(out)
    desired = ratios * n
    base = np.floor(desired).astype(int)
    remainder = n - base.sum()
    # give the +1's to largest fractional parts
    order = np.argsort(-(desired - base))
    targets = base.copy()
    targets[order[:remainder]] += 1

    names = np.array(["train", "val", "test"])
    counts = np.zeros(3, dtype=int)
    out["split"] = ""

    # 4) Greedy assignment of whole buckets to the split with largest shortfall
    for _, idxs in bucket_items:
        gaps = targets - counts
        dest = int(np.argmax(gaps)) if (gaps > 0).any() else int(np.argmin(counts))
        out.loc[idxs, "split"] = names[dest]
        counts[dest] += len(idxs)

    # 5) Slice and optionally drop helper columns
    cols_to_drop = ["_scaf"] if keep_split_col else ["_scaf", "split"]
    train_df = out[out["split"] == "train"].drop(columns=cols_to_drop, errors="ignore")
    val_df   = out[out["split"] == "val"  ].drop(columns=cols_to_drop, errors="ignore")
    test_df  = out[out["split"] == "test" ].drop(columns=cols_to_drop, errors="ignore")

    return train_df, val_df #, test_df
