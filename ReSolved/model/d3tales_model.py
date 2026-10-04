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
    """单目标模型：legacy 复现旧输入，conjugation 用于共轭特征对照。

    两种模式的层结构和权重形状一致，但输入语义不同，加载权重时须
    使用该次训练 config.json 记录的模式，不能仅凭权重形状判断兼容性。
    """
    def __init__(self, num_layers=7, emb_dim=128, bond_features="legacy"):
        if bond_features not in ("legacy", "conjugation"):
            raise ValueError(f"Unknown bond feature mode: {bond_features}")
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
        self.bond_features = bond_features

    def forward(self, data):
        # None values are internal placeholders for the inherited signature;
        # the single-target readout never consumes solvent information.
        if self.bond_features == "legacy":
            return super().forward(data, None, None)
        # 仅替换三个数值键输入中的第一项：重复的键类型改为是否共轭。
        # 键类型仍由 embedding 表示；环标记、环大小与输入维度保持不变。
        # 此处复用作者的模块和残差顺序，不修改原始图或作者源文件。
        atom_embed = self.atom_type_embedding(data.x[:, 0].long())
        h = self.lin_in_atoms(torch.cat([atom_embed, data.x[:, 1:]], dim=1))
        bond_embed = self.bond_type_embedding(data.edge_attr[:, 0].long())
        msg_e = self.lin_in_bonds(torch.cat([bond_embed, data.edge_attr[:, 1:]], dim=1))
        for conv in self.convs:
            # 节点和边均保留作者的逐层残差，确保对照仅涉及输入特征。
            h_next, msg_next = conv(h, data.edge_index, msg_e)
            h = h + h_next
            msg_e = msg_e + msg_next
        return self.readout(h, msg_e, data)
