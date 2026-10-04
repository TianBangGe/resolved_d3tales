import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_scatter import scatter


class MPNNLayer(MessagePassing):
    """
    Message Passing Layer following the standard PyG design:
    """

    def __init__(
        self,
        emb_dim: int,
        edge_dim: int,
        magic_number: int,
        magic_number_2: int,
        magic_number_3: int,
        magic_number_4: int,
        aggr='add'
    ):
        """
        Args:
            emb_dim (int): node embedding dimensionality.
            edge_dim (int): edge feature dimensionality.
            magic_number, magic_number_2, magic_number_3, magic_number_4 (int):
                custom hyperparameters to control layer widths.
            aggr (str): aggregation function: 'add', 'mean', or 'max'.
        """
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP used for message construction: \psi(h_i, h_j, e_ij)
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + edge_dim, magic_number * emb_dim),
            BatchNorm1d(magic_number * emb_dim),
            ReLU(),
            Linear(magic_number * emb_dim, magic_number_2 * emb_dim),
            ReLU(),
            Linear(magic_number_2 * emb_dim, emb_dim),
            ReLU()
        )

        # MLP used for node update: \phi(h_i, m_i)
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, magic_number_3 * emb_dim),
            BatchNorm1d(magic_number_3 * emb_dim),
            ReLU(),
            Linear(magic_number_3 * emb_dim, magic_number_4 * emb_dim),
            ReLU(),
            Linear(magic_number_4 * emb_dim, emb_dim),
            ReLU()
        )

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor):
        """
        One round of message passing: propagate node states and edge features.
        """
        out, msg_u = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out, msg_u

    def message(self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Construct messages m_ij = \psi(h_i, h_j, e_ij).
        h_i: dest node features
        h_j: source node features
        """
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs: Tensor, index: Tensor):
        """
        Aggregate messages from neighbors using `self.aggr`.
        """
        msg_u = inputs
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr), msg_u

    def update(self, aggr_out, h: Tensor):
        """
        Update node features using the aggregated messages.
        """
        agr_u, msg_u = aggr_out
        upd_in = torch.cat([h, agr_u], dim=-1)
        return self.mlp_upd(upd_in), msg_u

    def __repr__(self):
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
