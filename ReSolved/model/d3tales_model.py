"""ReSolved architecture migration: one graph readout and one direct target."""
import torch
from torch import nn
from torch_geometric.nn.aggr import SetTransformerAggregation

from .mpnn_model import MPNNModel


class ReductionReadout(nn.Module):
    def __init__(self, emb_dim, num_seed_points, heads):
        super().__init__()
        self.aggr = SetTransformerAggregation(emb_dim, num_seed_points, heads)
        self.aggr.reset_parameters()
        self.head = nn.Linear(num_seed_points * emb_dim, 1)

    def forward(self, h, msg_e, data, *_unused):
        # Identical node/edge token grouping to the author's SolvReadout.
        graph_vector = torch.cat([h, msg_e], dim=0)
        edge_batch = data.batch[data.edge_index[0]]
        combined_batch = torch.cat([data.batch, edge_batch], dim=-1)
        sorted_batch, indices = torch.sort(combined_batch, dim=0)
        return self.head(self.aggr(graph_vector[indices], sorted_batch))


class D3TaLESModel(MPNNModel):
    def __init__(self, num_layers=7, emb_dim=128):
        # Reuse the original constructor, embeddings, MPNN layers and forward.
        # The original task readout is replaced; none of its parameters remain.
        super().__init__(
            num_layers=num_layers, emb_dim=emb_dim,
            magic_number=2, magic_number_2=3,
            magic_number_3=1, magic_number_4=1,
            num_seed_points=3, heads=1,
            dim_atoms=8, dim_bond=6,
            emb_dielec=2, emb_refract=2, out_dim=1,
        )
        self.readout = ReductionReadout(emb_dim, num_seed_points=3, heads=1)

    def forward(self, data):
        # None values are internal placeholders for the inherited signature;
        # the single-target readout never consumes solvent information.
        return super().forward(data, None, None)
