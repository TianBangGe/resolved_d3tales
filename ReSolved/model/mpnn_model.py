import torch
from torch import nn
from torch.nn import Embedding, Linear

from .mpnn_layer import MPNNLayer
from .readout import SolvReadout


class MPNNModel(nn.Module):
    """
    High-level MPNN model that:
      1) Embeds atom & bond features
      2) Passes them through multiple MPNNLayer(s)
      3) Applies the final readout to incorporate solvent info & produce multi-task outputs
    """

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        magic_number: int,
        magic_number_2: int,
        magic_number_3: int,
        magic_number_4: int,
        num_seed_points: int,
        heads: int,
        dim_atoms: int,
        dim_bond: int,
        emb_dielec: int,
        emb_refract: int,
        out_dim: int = 1,
        num_atom_types: int = 120,  # e.g., len(ATOM_SYMBOLS)
        num_bond_types: int = 5
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.edge_dim = emb_dim

        # Atom & Bond Embeddings
        self.atom_type_embedding = Embedding(num_atom_types, dim_atoms)
        self.bond_type_embedding = Embedding(num_bond_types, dim_bond)

        # Linear input
        self.lin_in_atoms = Linear(8 + dim_atoms, emb_dim)
        self.lin_in_bonds = Linear(3 + dim_bond, emb_dim)

        # Build MPNN Layers
        self.convs = nn.ModuleList([
            MPNNLayer(
                emb_dim,
                self.edge_dim,
                magic_number,
                magic_number_2,
                magic_number_3,
                magic_number_4,
                aggr='add'
            )
            for _ in range(num_layers)
        ])

        # Readout block
        self.readout = SolvReadout(
            emb_dim=emb_dim,
            num_seed_points=num_seed_points,
            heads=heads,
            emb_dielec=emb_dielec,
            emb_refract=emb_refract,
            out_dim=out_dim
        )

    def forward(self, data, data_dielec, data_ref):
        # Node embedding
        atom_embed = self.atom_type_embedding(data.x[:, 0].long())
        other_atom_feats = data.x[:, 1:]
        x_cat = torch.cat([atom_embed, other_atom_feats], dim=1)
        h = self.lin_in_atoms(x_cat)

        # Edge embedding
        bond_embed = self.bond_type_embedding(data.edge_attr[:, 0].long())
        other_bond_feats = torch.cat(
            [data.edge_attr[:, :1], data.edge_attr[:, 2:5], data.edge_attr[:, 6:]], dim=1)
        msg_e = torch.cat([bond_embed, other_bond_feats], dim=1)
        msg_e = self.lin_in_bonds(msg_e)

        # Message passing (residual connections)
        for conv in self.convs:
            h_next, msg_next = conv(h, data.edge_index, msg_e)
            h = h + h_next
            msg_e = msg_e + msg_next

        # Readout
        return self.readout(h, msg_e, data, data_dielec, data_ref)
