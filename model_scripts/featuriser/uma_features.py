####################################################################################################
# To use the UMA models you must request access to the hugging face repository 
# (https://huggingface.co/facebook/UMA) and have logged in to Hugging Face using an access token
#
# huggingface-cli login
#
####################################################################################################

import argparse
from typing import Optional, List, Dict
import hashlib
import os
import pandas as pd
import tqdm

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets import data_list_collater

from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.io import write, read

import torch
import numpy as np

def smiles_to_ase(smiles: str | Chem.Mol, maxAttempts: int = 100) -> Optional[Atoms]:
    """
    Convert a SMILES string to an ASE Atoms object.
    Requires: RDKit and ASE.
    # TODO cache the 3D ASE files or load from file
    """
    # 1. Parse SMILES → RDKit Mol
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    mol = Chem.AddHs(mol)  # add hydrogens for 3D geometry

    # 2. Generate 3D coordinates
    res = -1
    for etkdg in [AllChem.ETKDGv3(), AllChem.ETKDGv2(), AllChem.ETKDG()]:
        res = AllChem.EmbedMolecule(mol, etkdg)
        if res == 0:
            break
    if res != 0:
        res = AllChem.EmbedMolecule(mol,useRandomCoords=True, maxAttempts=maxAttempts)
    
    if res != 0:
        Chem.RemoveStereochemistry(mol)
        res = AllChem.EmbedMolecule(mol,useRandomCoords=True, maxAttempts=maxAttempts)

    if res != 0:
        return None

    AllChem.UFFOptimizeMolecule(mol)  # geometry optimization

    # 3. Extract atomic symbols and positions
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    positions = [(p.x, p.y, p.z) for p in positions]

    # 4. Build ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms


def ase_to_atomicdata(atoms: Atoms, task_name: str) -> AtomicData:
    return AtomicData.from_ase(input_atoms=atoms, task_name=task_name)


def _hash_smiles(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


class AseCache:
    def __init__(self, cache_dir: str, max_attempts=100):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.max_attempts = max_attempts

        # In-memory store: {smiles_hash: Atoms}
        self.cache: Dict[str, Atoms] = {}

        # preload existing XYZ files into memory
        self._load_all_cached()

    def _file_for(self, smiles: str) -> str:
        return os.path.join(self.cache_dir, _hash_smiles(smiles) + ".xyz")

    def _load_all_cached(self):
        """Load all .xyz molecules in cache directory into RAM."""
        for file in os.listdir(self.cache_dir):
            if file.endswith(".xyz"):
                file_path = os.path.join(self.cache_dir, file)
                try:
                    atoms = read(file_path)
                    smiles_hash = file.replace(".xyz", "")
                    self.cache[smiles_hash] = atoms
                except Exception as e:
                    print(f"Failed to load {file} ({e}), skipping")


    def add_to_cache(self, smiles_lst: List[str]):
        for smiles in tqdm.tqdm(smiles_lst):
            smiles_hash = _hash_smiles(smiles)
            if smiles_hash not in self.cache:
                atoms = smiles_to_ase(smiles, maxAttempts=self.max_attempts)
                if atoms is None:
                    with open('failed.txt', 'a') as f:
                        f.write(f'{smiles}\n')
                    continue
                file_path = self._file_for(smiles)
                write(file_path, atoms)
                self.cache[smiles_hash] = atoms

    def __call__(self, smiles: str, generate: bool = False) -> Optional[Atoms]:
        atoms = None
        smiles_hash = _hash_smiles(smiles)
        # Return from memory if present
        if smiles_hash in self.cache:
            return self.cache[smiles_hash]
        # Else generate & store
        if generate:
            atoms = smiles_to_ase(smiles)
            write(self._file_for(smiles), atoms)
            self.cache[smiles_hash] = atoms
        return atoms

class UMAFeaturiser():
    def __init__(self, model_name, device, reduction: str ='mean', task_name: str = 'omol', ase_cache: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.reduction = reduction
        self.task_name = task_name

        self.model = self._load_model(model_name, device)

    def _load_model(self, model_name, device):
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        # Accounting for the use of averagemodel or hydramodel
        if hasattr(predictor, 'model'):
            if hasattr(predictor.model, 'module') and hasattr(predictor.model.module, 'backbone'):
                return predictor.model.module.backbone
            if hasattr(predictor.model, 'backbone'):
                return predictor.model.backbone

        raise ValueError('Model structure is not recognised.')
    
    def _reduce(self, x: torch.Tensor) -> np.ndarray:
        x = x.cpu().numpy()
        if self.reduction == 'mean':
            return x.mean(dim=0, keepdims=False)
        elif self.reduction is None or self.reduction == 'none':
            return x
        else:
            raise ValueError(f'Reduction ({self.reduction}) not recognised.')
    
    @torch.no_grad()
    def __call__(self, smiles: str | Chem.Mol) -> np.ndarray:
        # TODO Accept a batch of SMILES strings to process
        # SMILES to AtomData
        atom_data = ase_to_atomicdata(smiles_to_ase(smiles), task_name=self.task_name)
        batch = data_list_collater([atom_data], otf_graph=True)

        results = self.model(batch)
        embs = results['node_embedding']
        # Flatten the node embeddings
        # TODO Flatten according to the results['batch']
        embs = embs.view(-1, embs.shape[-1])

        return self._reduce(embs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='uma-s-1p1')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    featuriser = UMAFeaturiser(args.model_name, args.device)

    # data_path = '/home/ros282/data/polaris'
    # cache = AseCache('3d_cache', max_attempts=1000)
    # for fn in os.listdir(data_path):
    #     df = pd.read_csv(os.path.join(data_path, fn))
    #     smiles_lst = df['SMILES'].tolist()
    #     cache.add_to_cache(smiles_lst)

    smiles_lst = [
        'CC1=C(C)[N@@H+]([O-])[Co-5]2(C)([Co-5]34(C)(N([O-])C(C)=C(C)[N@H+]3[O-])[N+]([O-])=C(C)C(C)=[N+]4[O-])(N1[O-])[N+]([O-])=C(C)C(C)=[N+]2[O-]',
        'CC1=C2CC(C=O)C(C1)C(C(C)C)C2',
        'c1cc2ccc1CCc1ccc(cc1)CC2',
        'O=C(CCCN1CCC(O)(c2cc3ccc2CCc2ccc(cc2)CC3)CC1)c1cc2ccc1CCc1ccc(cc1)CC2',
        'O=C(NCCCCN1CCN(c2cc3ccc2CCc2ccc(cc2)CC3)CC1)c1cc2ccc1CCc1ccc(cc1)CC2',
        'O=C(CCCN1CCC(c2cc3ccc2CCc2ccc(cc2)CC3)CC1)c1cc2ccc1CCc1ccc(cc1)CC2',
        'CCC(C)=C1Oc2ccc(F)cc2-c2ccc3c(c21)C(C)=CC(C)(C)N3',
    ]
    smiles_lst = list(set(smiles_lst))
    cache = AseCache('3d_cache', max_attempts=1_000_000)
    cache.add_to_cache(smiles_lst)


    err_count = 0
    with tqdm.tqdm(smiles_lst, desc="Featurising") as pbar:
        for smiles in pbar:
            try:
                embeds = featuriser(smiles)
            except ValueError as e:
                err_count += 1
            pbar.set_postfix(errs=err_count)
    print(err_count)

    featuriser('CC[C@H]1[C@@H]2C[C@H]3[C@@H]4N(C)c5ccccc5[C@@]54C[C@@H]([C@H]2[C@@H]5O)[N@]3[C@@H]1O')
    