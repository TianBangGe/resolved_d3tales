import torch
import numpy as np
from rdkit import Chem


ATOM_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
    'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti',
    'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
    'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
    'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
    'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Rf', 'Db', 'Sg',
    'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Fl', 'Lv', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
]

NUM_ATOM_TYPES = len(ATOM_SYMBOLS)
NUM_BOND_TYPES = 5  # [SINGLE, DOUBLE, TRIPLE, AROMATIC, UNSPECIFIED]
NUM_STEREO_TYPES = 4


def categorical_type(x, permitted_list):
    """
    Converts a categorical feature into an integer index based on a permitted list.
    """
    return permitted_list.index(x)


def get_ring_size(atom_or_bond, max_size=12) -> int:
    """
    Returns the ring size if the atom_or_bond is in a ring, 0 otherwise.
    Looks up to `max_size` rings.
    """
    if not atom_or_bond.IsInRing():
        return 0
    for i in range(max_size):
        if atom_or_bond.IsInRingSize(i):
            return i
    return 0


def get_atom_features(atom, use_chirality=True, hydrogens_implicit=False) -> torch.Tensor:
    """
    Converts an RDKit atom object into a PyTorch tensor of atom features.
    """
    permitted_list_of_atoms = ATOM_SYMBOLS
    atom_type_enc = categorical_type(
        str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = atom.GetDegree()
    is_in_a_ring_enc = atom.IsInRing()
    r_s = get_ring_size(atom)
    is_aromatic_enc = atom.GetIsAromatic()
    atomic_mass_scaled = (atom.GetMass() - 10.812) / 116.092
    vdw_radius_scaled = (Chem.GetPeriodicTable().GetRvdw(
        atom.GetAtomicNum()) - 1.5) / 0.6
    covalent_radius_scaled = (Chem.GetPeriodicTable(
    ).GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76
    valence = atom.GetTotalValence()

    if hydrogens_implicit:
        n_hydrogens_enc = atom.GetTotalNumHs()
        # Potentially add to feature vector if needed.

    atom_features = torch.tensor(
        [
            atom_type_enc,
            r_s,
            is_in_a_ring_enc,
            n_heavy_neighbors_enc,
            is_aromatic_enc,
            atomic_mass_scaled,
            vdw_radius_scaled,
            covalent_radius_scaled,
            valence
        ]
    )
    return atom_features


def get_bond_features(bond, use_stereochemistry=False) -> torch.Tensor:
    """
    Converts an RDKit bond object into a PyTorch tensor of bond features.
    """
    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        Chem.rdchem.BondType.UNSPECIFIED
    ]
    bond_type_enc = categorical_type(
        bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = bond.GetIsConjugated()
    bond_is_in_ring_enc = bond.IsInRing()
    bond_ring_size = get_ring_size(bond)

    if use_stereochemistry:
        stereo_type_enc = categorical_type(
            str(bond.GetStereo()),
            ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        )

    bond_features = torch.tensor(
        [
            bond_type_enc,
            bond_is_conj_enc,
            bond_is_in_ring_enc,
            bond_ring_size
        ]

    )
    return bond_features
