import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_max_pool


class GATv2(torch.nn.Module):
    def __init__(
        self, input_feat_dim, dim_shapes, heads, num_layers, num_classes, dropout_p
    ):

        super(GATv2, self).__init__()
        assert num_layers >= 1, "Number of layers should be more than or equal to 1"
        self.num_layers = num_layers
        self.linear = None

        if input_feat_dim != dim_shapes[0][0]:
            self.linear = nn.Linear(input_feat_dim, dim_shapes[0][0])

        self.convs = nn.ModuleList()
        for l in range(num_layers):
            if l == 0:
                self.convs.append(
                    GATv2Conv(dim_shapes[l][0], dim_shapes[l][1], heads=heads)
                )
            else:
                self.convs.append(
                    GATv2Conv(dim_shapes[l][0] * heads, dim_shapes[l][1], heads=heads)
                )

        self.dropout = dropout_p
        self.pooling = global_max_pool

        self.classifier = nn.Sequential(
            nn.Linear(heads * dim_shapes[-1][1], 16),
            nn.Linear(16, num_classes),
        )

    def forward(self, batched_data):

        x, edge_index, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.batch,
        )

        if self.linear is not None:
            x = self.linear(x.float())

        for l in range(self.num_layers):
            x = F.relu(self.convs[l](x.float(), edge_index))

        x = self.pooling(x, batch)
        # x = F.dropout(x, p=self.dropout)
        x = self.classifier(x)

        return x
