import torch
from torch import nn
from torch_geometric.nn.aggr import SetTransformerAggregation


class SolvReadout(nn.Module):
    """
    A custom readout block that uses:
      - Contatanation of nodes/edges
      - SetTransformerAggregation for node/edge embeddings
      - Additional linear layers to incorporate solvent embeddings
    """

    def __init__(
        self,
        emb_dim: int,
        num_seed_points: int,
        heads: int,
        emb_dielec: int,
        emb_refract: int,
        out_dim: int
    ):
        super().__init__()

        # Two separate set transformer aggregations for EA and solvent embeddings
        self.aggr_ea = SetTransformerAggregation(
            emb_dim, num_seed_points, heads)
        self.aggr_ea.reset_parameters()

        self.aggr_solv = SetTransformerAggregation(
            emb_dim, num_seed_points, heads)
        self.aggr_solv.reset_parameters()

        # Embeddings for solvent properties
        self.emb_dielec_lin = nn.Linear(1, emb_dielec)
        self.emb_refract_lin = nn.Linear(1, emb_refract)

        # Linear layers for final prediction
        self.lin_pred_ea = nn.Linear(num_seed_points * emb_dim, out_dim)
        hidden_dim = num_seed_points * emb_dim + emb_dielec + emb_refract
        self.lin_pred_solv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h, msg_e, data, data_dielec, data_ref):
        """
        h, msg_e: node and edge features post message passing
        data: original PyG data object
        data_dielec, data_ref: dielectric constan and refraction coefficent inputs for each solvent
        """
        data_dielec = data_dielec.unsqueeze(-1)
        data_ref = data_ref.unsqueeze(-1)

        # Combine node/edge features
        graph_vector = torch.cat([h, msg_e], dim=0)

        # Here we grouping nodes and edges by graph, since  edges are not directly labeled with
        # a batch index. We obtain the batch index of an edge by looking at the batch index of
        # its “row” node.
        row, col = data.edge_index
        edge_batch = data.batch[row]
        combined_batch = torch.cat([data.batch, edge_batch], dim=-1)

        # Sort for stable grouping by batch
        sorted_batch, indices = torch.sort(combined_batch, dim=0)
        sorted_feats = graph_vector[indices]

        # Aggregate for EA
        graph_vector_ea = self.aggr_ea(sorted_feats, sorted_batch)

        # Aggregate for solv
        graph_vector_solv = self.aggr_solv(sorted_feats, sorted_batch)

        # Embeddings for solvent descriptors
        disp_emb = self.emb_dielec_lin(data_dielec)
        ref_emb = self.emb_refract_lin(data_ref)

        ea_pred = self.lin_pred_ea(graph_vector_ea)
        solv_pred = self.lin_pred_solv(
            torch.cat([graph_vector_solv, disp_emb, ref_emb], dim=1))

        # final output shape => [-ea, -ea + solv_pred]
        return torch.cat([-ea_pred, -ea_pred + solv_pred], dim=1)
