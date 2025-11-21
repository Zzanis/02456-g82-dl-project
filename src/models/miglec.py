from enum import StrEnum
from typing import Type

import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool
from torch_geometric.nn.conv import GraphConv


class NormLayerType(StrEnum):
    """Normalisation layer types with associated constructors"""

    BATCH = "batch"
    LAYER = "layer"
    NONE = "none"

    def __new__(cls, value: str) -> "NormLayerType":
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    @property
    def new(self) -> Type[torch.nn.Module]:
        """Return the constructor for this layer type"""

        layer_map = {
            self.BATCH: BatchNorm,
            self.LAYER: LayerNorm,
            self.NONE: torch.nn.Identity,
        }
        return layer_map[self]


class GCN_LC(torch.nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int = 64):
        super(GCN_LC, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        dropout: float = 0,
        dropout_final: float = 0,
        use_residual: bool = False,
        norm_layer_type: NormLayerType = NormLayerType.NONE,
    ):
        super(GCN, self).__init__()
        self.p_dropout = dropout
        self.p_dropout_final = dropout_final
        self.use_residual = use_residual

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.norm1 = norm_layer_type.new(hidden_channels)
        self.norm2 = norm_layer_type.new(hidden_channels)
        self.norm3 = norm_layer_type.new(hidden_channels)
        self.norm4 = norm_layer_type.new(hidden_channels)
        self.norm5 = norm_layer_type.new(hidden_channels)
        self.residual1 = torch.nn.Linear(hidden_channels, hidden_channels)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x_res = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        if self.use_residual:
            x = x + self.residual1(x_res)
        x = self.conv4(x, edge_index)
        x = self.norm4(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv5(x, edge_index)
        x = self.norm5(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv6(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.p_dropout_final, training=self.training)
        x = self.mlp(x)

        return x


class GCN_GraphConv(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        dropout: float = 0,
        dropout_final: float = 0,
        use_residual: bool = False,
        norm_layer_type: NormLayerType = NormLayerType.NONE,
    ):
        super(GCN_GraphConv, self).__init__()
        self.p_dropout = dropout
        self.p_dropout_final = dropout_final
        self.use_residual = use_residual

        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(hidden_channels, hidden_channels)
        self.conv6 = GraphConv(hidden_channels, hidden_channels)
        self.norm1 = norm_layer_type.new(hidden_channels)
        self.norm2 = norm_layer_type.new(hidden_channels)
        self.norm3 = norm_layer_type.new(hidden_channels)
        self.norm4 = norm_layer_type.new(hidden_channels)
        self.norm5 = norm_layer_type.new(hidden_channels)
        self.residual1 = torch.nn.Linear(hidden_channels, hidden_channels)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x_res = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        if self.use_residual:
            x = x + self.residual1(x_res)
        x = self.conv4(x, edge_index)
        x = self.norm4(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv5(x, edge_index)
        x = self.norm5(x)
        x = x.relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv6(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.p_dropout_final, training=self.training)
        x = self.mlp(x)

        return x
