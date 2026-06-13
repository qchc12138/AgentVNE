import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np


class ColumnWiseTensorNetwork(nn.Module):
    """Column-wise neural tensor network: computes Hj * W_j * Hi^T for each SN node j.

    Args:
        hidden_dim: feature dimension of both graphs
        num_nodes_j: fixed number of SN nodes (must match at inference)

    Input:  Hi [N_vn, hidden_dim], Hj [N_sn, hidden_dim]
    Output: Z  [N_sn, N_vn]
    """

    def __init__(self, hidden_dim, num_nodes_j):
        super(ColumnWiseTensorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes_j = num_nodes_j
        self.W = nn.Parameter(torch.randn(num_nodes_j, hidden_dim, hidden_dim))

    def forward(self, Hi, Hj):
        N2 = Hj.size(0)
        if N2 != self.num_nodes_j:
            raise ValueError(
                f"N2={N2} does not match num_nodes_j={self.num_nodes_j}. "
                f"Ensure target graph node count is consistent."
            )
        selected_W = self.W
        hj_expanded = Hj.unsqueeze(1)
        hj_w = torch.matmul(hj_expanded, selected_W).squeeze(1)
        Z = torch.matmul(hj_w, Hi.transpose(0, 1))
        return Z


class SimuVNE(nn.Module):
    """SimuVNE main neural network model.

    GCN encoder -> Transformer encoder -> NTN similarity ->
    Transformer refinement -> sigmoid normalisation.
    """

    def __init__(self, input_dim=6, hidden_dim=64, hist_dim=32, num_nodes_j=10):
        super(SimuVNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hist_dim = hist_dim
        self.num_nodes_j = num_nodes_j

        # GCN for graph i (VN)
        self.gcn1_i = GCNConv(input_dim, hidden_dim)
        self.gcn2_i = GCNConv(hidden_dim, hidden_dim)

        # GCN for graph j (SN)
        self.gcn1_j = GCNConv(input_dim, hidden_dim)
        self.gcn2_j = GCNConv(hidden_dim, hidden_dim)

        # 1-layer Transformer encoders for VN and SN
        encoder_layer_i = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=1, dim_feedforward=256,
            dropout=0.1, batch_first=False,
        )
        self.encoder_i = nn.TransformerEncoder(encoder_layer_i, num_layers=1)

        encoder_layer_j = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=1, dim_feedforward=256,
            dropout=0.1, batch_first=False,
        )
        self.encoder_j = nn.TransformerEncoder(encoder_layer_j, num_layers=1)

        # Column-wise NTN
        self.ntn = ColumnWiseTensorNetwork(hidden_dim, num_nodes_j=num_nodes_j)

        # 3-layer Transformer on NTN output (d_model = num_sn_nodes)
        encoder_layer_z = nn.TransformerEncoderLayer(
            d_model=num_nodes_j, nhead=1, dim_feedforward=256,
            dropout=0.1, batch_first=False,
        )
        self.encoder_z = nn.TransformerEncoder(encoder_layer_z, num_layers=3)

        self.dropout = nn.Dropout(0.1)

    def calculate_histogram(self, S, bins=32):
        """Compute histogram features of similarity matrix S.

        Args:
            S: [N1, N2] similarity matrix
            bins: number of histogram bins

        Returns:
            hist_features: [bins] histogram feature vector
        """
        S_flat = S.view(-1)
        hist = torch.histc(S_flat, bins=bins, min=0.0, max=1.0)
        hist = hist / (hist.sum() + 1e-8)
        return hist

    def forward(self, data_i, data_j):
        """Forward pass.

        Args:
            data_i: graph i (VN) -- PyG Data with x [N1,6], edge_index [2,E1]
            data_j: graph j (SN) -- PyG Data with x [N2,6], edge_index [2,E2]

        Returns:
            [N1, N2] row-normalised matching scores
        """
        x_i, edge_index_i = data_i.x, data_i.edge_index
        x_j, edge_index_j = data_j.x, data_j.edge_index

        # GCN encoding
        U_i = F.relu(self.gcn1_i(x_i, edge_index_i))
        U_i = self.dropout(U_i)
        U_i = F.relu(self.gcn2_i(U_i, edge_index_i))

        U_j = F.relu(self.gcn1_j(x_j, edge_index_j))
        U_j = self.dropout(U_j)
        U_j = F.relu(self.gcn2_j(U_j, edge_index_j))

        # Initial similarity matrix S [N1, N2]
        S = torch.sigmoid(torch.matmul(U_i, U_j.transpose(0, 1)))
        hist_S = self.calculate_histogram(S, bins=self.hist_dim)
        self.last_histogram = hist_S

        # Transformer encoding of GCN features
        U_i_input = U_i.unsqueeze(1)
        Hi_encoded = self.encoder_i(U_i_input)
        Hi = Hi_encoded.squeeze(1)

        U_j_input = U_j.unsqueeze(1)
        Hj_encoded = self.encoder_j(U_j_input)
        Hj = Hj_encoded.squeeze(1)

        # NTN interaction
        Z = self.ntn(Hi, Hj)
        Z = Z.transpose(0, 1)

        # Refinement Transformer on NTN output
        Z_input = Z.unsqueeze(1)
        Z_encoded = self.encoder_z(Z_input)
        Z_prime = Z_encoded.squeeze(1)

        # Sigmoid + row-normalise -> valid probability distribution
        Z_normalized = torch.sigmoid(Z_prime)
        output = Z_normalized / (Z_normalized.sum(dim=1, keepdim=True) + 1e-8)

        return output


def create_model(input_dim=6, hidden_dim=64, hist_dim=32, num_nodes_j=10):
    """Create model instance."""
    model = SimuVNE(input_dim=input_dim, hidden_dim=hidden_dim, hist_dim=hist_dim, num_nodes_j=num_nodes_j)
    return model


if __name__ == "__main__":
    from torch_geometric.data import Data

    # diagram i: 5 nodes
    x_i = torch.randn(5, 6)
    edge_index_i = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data_i = Data(x=x_i, edge_index=edge_index_i)

    # diagram j: 4 nodes
    x_j = torch.randn(4, 6)
    edge_index_j = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data_j = Data(x=x_j, edge_index=edge_index_j)

    model = create_model(num_nodes_j=4)
    output = model(data_i, data_j)
    print(f"output shape: {output.shape}")
    print(f"output:\n{output}")
