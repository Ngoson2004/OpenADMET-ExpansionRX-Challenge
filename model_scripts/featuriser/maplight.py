import numpy as np
from sklearn import preprocessing

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP
from rdkit.Chem import rdReducedGraphs, MACCSkeys

# from molfeat.trans.pretrained import PretrainedDGLTransformer
from chemprop.featurizers.molecule import (
    MorganBinaryFeaturizer,
    MorganCountFeaturizer,
    V1RDKit2DNormalizedFeaturizer,
)
from skfp.fingerprints import ERGFingerprint, MACCSFingerprint
from featuriser.mordred_recursive import get_mordred
from featuriser.padel import get_padel_df 
# from cddd.inference import InferenceModel  # For CDDD descriptors

class scaler:
    def __init__(self, log=False):
        self.log = log
        self.offset = None
        self.scaler = None

    def fit(self, y):
        # make the values non-negative
        self.offset = np.min([np.min(y), 0.0])
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        self.scaler = preprocessing.StandardScaler().fit(y)

    def transform(self, y):
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        y_scale = self.scaler.transform(y)

        return y_scale

    def inverse_transform(self, y_scale):
        y = self.scaler.inverse_transform(y_scale.reshape(-1, 1))

        if self.log:
            y = 10.0**y - 1.0

        y = y + self.offset

        return y


# from https://github.com/rdkit/rdkit/discussions/3863
def count_to_array(fingerprint):
    # For count fingerprints, we need to get the non-zero elements and their counts
    # array = np.zeros((2048,), dtype=np.int8)
    return np.array(fingerprint.ToList(), dtype=np.float32)


def get_avalon_fingerprints(molecules, count=False, n_bits=2048):
    # Handle single molecule vs list of molecules
    # if not count and not isinstance(molecules, list):
    #     molecules = [molecules]
    
    fingerprints = []
    # for x in molecules:
    fp = GetAvalonCountFP(molecules, nBits=n_bits) if count else GetAvalonFP(molecules, nBits=n_bits)
    fp = count_to_array(fp)
    fingerprints.append(fp)
    
    result = np.stack(fingerprints)
    
    return result.flatten()


def get_morgan_fingerprints(molecules, count=False):
    featurizer = MorganCountFeaturizer() if count else MorganBinaryFeaturizer()
    return featurizer(molecules)
  
def get_erg_fingerprints(molecules, count=False):
    # Handle single molecule vs list of molecules
    if not count and not isinstance(molecules, list):
        molecules = [molecules]
    
    fingerprints = []
    # for mol in molecules:
    fp = rdReducedGraphs.GetErGFingerprint(molecules) if count else ERGFingerprint(variant="bit").transform(molecules)
    fingerprints.append(fp)

    result = np.stack(fingerprints)

    return result.flatten()

def get_rdkit_features(molecules):
    featurizer = V1RDKit2DNormalizedFeaturizer()
    return featurizer(molecules)

def get_maccs_features(molecules, count=False):
    # Handle single molecule vs list of molecules
    if count and not isinstance(molecules, list):
        molecules = [molecules]
    
    fingerprints = []
    # for mol in molecules:
    fp = MACCSFingerprint(count=True).transform(molecules) if count else MACCSkeys.GenMACCSKeys(molecules)
    # Convert to numpy array and ensure it's 1D
    # fp_array = count_to_array(fp) if not count else fp
    fingerprints.append(fp)

    result = np.stack(fingerprints)

    return result.flatten()

# def get_mordred_features(molecules):
#     featuriser = get_mordred_fast()
#     return featuriser(molecules)

def get_maplight(molecules, count=False, use3d=False):
    RDLogger.DisableLog('rdApp.*')
    
    # Convert SMILES string to Mol object if needed
    if isinstance(molecules, str):
        molecules = Chem.MolFromSmiles(molecules)
    
    # Check if we're dealing with a single molecule
    is_single = not isinstance(molecules, list)
    
    fingerprints = []

    fingerprints.append(get_morgan_fingerprints(molecules, count=count))
    fingerprints.append(get_avalon_fingerprints(molecules, count=count))
    fingerprints.append(get_erg_fingerprints(molecules, count=count))
    fingerprints.append(get_maccs_features(molecules, count=count)) 
    if count: 
        fingerprints.append(get_rdkit_features(molecules))
    if use3d:
        fingerprints.append(get_padel_df(molecules))

    if is_single:
        # For single molecule, concatenate 1D arrays
        result = np.concatenate(fingerprints)
    else:
        # For multiple molecules, concatenate along axis 1
        result = np.concatenate(fingerprints, axis=1)
    
    # Final validation: replace any remaining NaN or infinite values
    result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
    return result

'''
def get_gin_supervised_masking(molecules):
    # Handle single molecule vs list of molecules
    if not isinstance(molecules, list):
        molecules = [molecules]
    
    transformer = PretrainedDGLTransformer(kind='gin_supervised_masking', dtype=float)
    result = transformer(molecules)
    
    # Replace NaN and infinite values with zeros
    # result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If input was single molecule, return 1D array
    if len(molecules) == 1:
        return result.flatten()
    return result

def get_fingerprints_gnn(molecules):
    RDLogger.DisableLog('rdApp.*')
    
    # Convert SMILES string to Mol object if needed
    if isinstance(molecules, str):
        molecules = Chem.MolFromSmiles(molecules)
    
    # Check if we're dealing with a single molecule
    is_single = not isinstance(molecules, list)
    

    fingerprints = []

    fingerprints.append(get_morgan_fingerprints(molecules))
    fingerprints.append(get_avalon_fingerprints(molecules))
    fingerprints.append(get_erg_fingerprints(molecules))
    fingerprints.append(get_rdkit_features(molecules))
    fingerprints.append(get_padel_features(molecules))
    fingerprints.append(get_mordred_features(molecules))
    fingerprints.append(get_maccs_features(molecules))
    fingerprints.append(get_gin_supervised_masking(molecules))

    if is_single:
        # For single molecule, concatenate 1D arrays
        result = np.concatenate(fingerprints)
    else:
        # For multiple molecules, concatenate along axis 1
        result = np.concatenate(fingerprints, axis=1)
    
    # Final validation: replace any remaining NaN or infinite values
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
#     return result
'''

