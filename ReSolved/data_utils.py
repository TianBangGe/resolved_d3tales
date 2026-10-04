import random
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Optional
from rdkit import Chem
from features import get_atom_features, get_bond_features


def set_seed(seed: int):
    """
    Set the random seed for reproducibility across multiple modules.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)


def smi_to_pyg(
    smi: str,
    EA: float,
    RP_EV_ACN: float,
    RP_water: float,
    RP_THF: float,
    RP_DMSO: float,
    RP_DMF: float,
    device: torch.device,
) -> Optional[Data]:
    """
    Converts a SMILES string to a PyG `Data` object with atom and bond features.
    """
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if mol is None:
        return None

    atom_features = [get_atom_features(a, device) for a in mol.GetAtoms()]

    edge_indices = []
    bond_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        bond_features.append(get_bond_features(bond))
        edge_indices.append((j, i))
        bond_features.append(get_bond_features(bond))

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
    edge_attr = torch.stack(bond_features, dim=0)
    x = torch.stack(atom_features, dim=0)

    # Stack atom features into a single [num_atoms, feature_dim] tensor

    data = Data(
        edge_index=edge_index,
        x=x,
        edge_attr=edge_attr,
        # y: The first value is -EA, the subsequent ones are Reduction Potentials in different solvents
        y=torch.FloatTensor(
            [[-EA, RP_EV_ACN, RP_water, RP_THF, RP_DMSO, RP_DMF]]),
        # we hardcode the solvent parameters, since every datapoint is computed in same 5 solvents,
        # but ideally we should made it more customisable
        solv_dielec=torch.FloatTensor([[20.7, 80.4, 7.25, 47.2, 38.3]]),
        solv_refract=torch.FloatTensor([[1.359, 1.33, 1.407, 1.479, 1.430]]),
        mol=mol,
        smiles=smi
    )
    return data.to(device)


class SolvDataset(Dataset):
    """
    Custom dataset class that converts a list of SMILES into PyG `Data` objects.
    """

    def __init__(
        self,
        smiles,
        EA,
        RP_ev_ACN,
        RP_water,
        RP_THF,
        RP_DMSO,
        RP_DMF,
        device
    ):
        # Since the dataset is small, we process everything during the initialisation
        mols = [
            smi_to_pyg(smi, ea, acn, h2o, thf, dmso, dmf, device)
            for smi, ea, acn, h2o, thf, dmso, dmf in zip(
                smiles, EA, RP_ev_ACN, RP_water, RP_THF, RP_DMSO, RP_DMF
            )
        ]
        # Filtering out None objects (failed SMILES)
        self.X = [m for m in mols if m is not None]

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)
