"""Author's graph encoding with fixed ID splits and sealed test targets."""
import csv
import math

import torch
from rdkit import Chem
from torch_geometric.data import Data

from features import get_atom_features, get_bond_features


def molecule_graph(dataset_id, smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for {dataset_id}: {smiles}")
    if Chem.GetFormalCharge(mol) != 0:
        raise ValueError(f"Non-neutral molecule: {dataset_id}")
    if not math.isfinite(target):
        raise ValueError(f"Non-finite target for {dataset_id}")
    mol = Chem.AddHs(mol)
    x = torch.stack([get_atom_features(atom) for atom in mol.GetAtoms()])
    edges, features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
        features.extend([get_bond_features(bond), get_bond_features(bond)])
    if not edges:
        raise ValueError(f"No bonds for {dataset_id}; unsupported by author's encoder")
    return Data(
        x=x, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(features), y=torch.tensor([[target]], dtype=torch.float32),
        dataset_id=dataset_id, smiles=smiles,
    )


def load_development_graphs(data_path, split_path, train_limit=None, valid_limit=None):
    limits = {"train": train_limit, "valid": valid_limit}
    if any(limit is not None and limit < 1 for limit in limits.values()):
        raise ValueError("Development graph limits must be positive")
    assignments = {}
    with open(split_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["dataset_id"]
            if key in assignments:
                raise ValueError(f"Duplicate split dataset_id: {key}")
            if row["split"] not in {"train", "valid", "test"}:
                raise ValueError(f"Unknown split for {key}: {row['split']}")
            assignments[key] = row
    graphs = {"train": [], "valid": []}
    seen = set()
    with open(data_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["dataset_id"]
            if key in seen:
                raise ValueError(f"Duplicate data dataset_id: {key}")
            seen.add(key)
            if key not in assignments:
                raise ValueError(f"Missing split assignment: {key}")
            assignment = assignments[key]
            # Reject test rows before interpreting labels or building graphs.
            if assignment["split"] == "test":
                continue
            group = assignment["split"]
            if limits[group] is not None and len(graphs[group]) >= limits[group]:
                continue
            if row["canonical_smiles"] != assignment["canonical_smiles"]:
                raise ValueError(f"SMILES mismatch for {key}")
            if float(row["groundState_charge"]) != 0:
                raise ValueError(f"Non-neutral groundState_charge for {key}")
            graph = molecule_graph(key, row["canonical_smiles"], float(row["reduction_potential"]))
            graphs[assignment["split"]].append(graph)
            if sum(map(len, graphs.values())) % 5000 == 0:
                print(f"Encoded {sum(map(len, graphs.values()))} development graphs", flush=True)
    if seen != set(assignments):
        raise ValueError("Data and split dataset_id coverage differ")
    if not graphs["train"] or not graphs["valid"]:
        raise ValueError("Train and Validation must both be nonempty")
    return graphs["train"], graphs["valid"]
