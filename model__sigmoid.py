import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np



class SelfAttention(nn.Module):
    """鑷敞鎰忓姏鏈哄埗灞?""
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        """
        x: [N, hidden_dim] - 鑺傜偣鐗瑰緛
        """
        Q = self.query(x)  # [N, hidden_dim]
        K = self.key(x)    # [N, hidden_dim]
        V = self.value(x)  # [N, hidden_dim]
        
        # 璁＄畻娉ㄦ剰鍔涙潈閲?
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [N, N]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 搴旂敤娉ㄦ剰鍔涙潈閲?
        output = torch.matmul(attention_weights, V)  # [N, hidden_dim]
        return output




class ColumnWiseTensorNetwork(nn.Module):
    """閫愬垪绁炵粡寮犻噺缃戠粶锛屽疄鐜?hj * Wj * Hi^T"""
    def __init__(self, hidden_dim, num_nodes_j):
        super(ColumnWiseTensorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes_j = num_nodes_j
        self.W = nn.Parameter(torch.randn(num_nodes_j, hidden_dim, hidden_dim))
    
    def forward(self, Hi, Hj):
        """
        Hi: [N1, hidden_dim]
        Hj: [N2, hidden_dim]
        杩斿洖: Z [N2, N1]
        """
        N2 = Hj.size(0)
        if N2 != self.num_nodes_j:
            raise ValueError(
                f"N2={N2} 涓庤瀹氱殑 num_nodes_j={self.num_nodes_j} 涓嶄竴鑷达紝璇风‘淇濈洰鏍囧浘鑺傜偣鏁伴噺鎭掑畾銆?
            )
        selected_W = self.W
        hj_expanded = Hj.unsqueeze(1)
        hj_w = torch.matmul(hj_expanded, selected_W).squeeze(1)
        Z = torch.matmul(hj_w, Hi.transpose(0, 1))
        return Z


class SimuVNE(nn.Module):
    """涓昏鐨凷imuVNE绁炵粡缃戠粶妯″瀷"""
    def __init__(self, input_dim=6, hidden_dim=64, hist_dim=32, num_nodes_j=10):
        super(SimuVNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hist_dim = hist_dim  # 鐩存柟鍥剧殑bins鏁伴噺
        self.num_nodes_j = num_nodes_j
        
        # GCN灞?- 涓哄浘Gi鍗曠嫭鐨勭綉缁?
        self.gcn1_i = GCNConv(input_dim, hidden_dim)
        self.gcn2_i = GCNConv(hidden_dim, hidden_dim)
        
        # GCN灞?- 涓哄浘Gj鍗曠嫭鐨勭綉缁?
        self.gcn1_j = GCNConv(input_dim, hidden_dim)
        self.gcn2_j = GCNConv(hidden_dim, hidden_dim)
        
        # Encoder灞?- 涓哄浘Gi鍜孏j浣跨敤1灞俆ransformer Encoder
        encoder_layer_i = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=1,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=False
        )
        self.encoder_i = nn.TransformerEncoder(encoder_layer_i, num_layers=1)
        
        encoder_layer_j = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=1,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=False
        )
        self.encoder_j = nn.TransformerEncoder(encoder_layer_j, num_layers=1)
        
        # 鏂扮殑閫愬垪NTN涓巈ncoder锛堜娇鐢≒yTorch瀹樻柟TransformerEncoderLayer锛?灞傦級
        self.ntn = ColumnWiseTensorNetwork(hidden_dim, num_nodes_j=num_nodes_j)
        encoder_layer_z = nn.TransformerEncoderLayer(
            d_model=num_nodes_j,
            nhead=1,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=False  # 浣跨敤 [seq_len, batch_size, d_model] 鏍煎紡
        )
        self.encoder_z = nn.TransformerEncoder(encoder_layer_z, num_layers=3)
        
        self.dropout = nn.Dropout(0.1)
    
    def calculate_histogram(self, S, bins=32):
        """
        璁＄畻鐩镐技搴︾煩闃礢鐨勭洿鏂瑰浘鐗瑰緛
        
        Args:
            S: [N1, N2] - 鐩镐技搴︾煩闃?
            bins: int - 鐩存柟鍥剧殑鍒嗙鏁伴噺
        
        Returns:
            hist_features: [bins] - 鐩存柟鍥剧壒寰佸悜閲?
        """
        S_flat = S.view(-1)
        hist = torch.histc(S_flat, bins=bins, min=0.0, max=1.0)
        hist = hist / (hist.sum() + 1e-8)
        
        return hist
        
    def forward(self, data_i, data_j):
        """
        data_i: 鍥綠i鐨勬暟鎹?(x_i: [N1, 6], edge_index_i: [2, E1])
        data_j: 鍥綠j鐨勬暟鎹?(x_j: [N2, 6], edge_index_j: [2, E2])
        杩斿洖: [N1, N2] - 姣忎釜鑺傜偣瀵圭殑鍖归厤鍒嗘暟
        """
        x_i, edge_index_i = data_i.x, data_i.edge_index
        x_j, edge_index_j = data_j.x, data_j.edge_index
        
        U_i = F.relu(self.gcn1_i(x_i, edge_index_i))
        U_i = self.dropout(U_i)
        U_i = F.relu(self.gcn2_i(U_i, edge_index_i))
        
        U_j = F.relu(self.gcn1_j(x_j, edge_index_j))
        U_j = self.dropout(U_j)
        U_j = F.relu(self.gcn2_j(U_j, edge_index_j))
        
        S = torch.sigmoid(torch.matmul(U_i, U_j.transpose(0, 1)))
        hist_S = self.calculate_histogram(S, bins=self.hist_dim)
        self.last_histogram = hist_S
        
        U_i_input = U_i.unsqueeze(1)
        Hi_encoded = self.encoder_i(U_i_input)
        Hi = Hi_encoded.squeeze(1)
        
        U_j_input = U_j.unsqueeze(1)
        Hj_encoded = self.encoder_j(U_j_input)
        Hj = Hj_encoded.squeeze(1)
        
        Z = self.ntn(Hi, Hj)
        Z = Z.transpose(0, 1)
        
        Z_input = Z.unsqueeze(1)
        Z_encoded = self.encoder_z(Z_input)
        Z_prime = Z_encoded.squeeze(1)
        
        Z_normalized = torch.sigmoid(Z_prime)
        output = Z_normalized / (Z_normalized.sum(dim=1, keepdim=True) + 1e-8)
        
        return output


def create_model(input_dim=6, hidden_dim=64, hist_dim=32, num_nodes_j=10):
    """鍒涘缓妯″瀷瀹炰緥"""
    model = SimuVNE(input_dim=input_dim, hidden_dim=hidden_dim, hist_dim=hist_dim)
    return model


if __name__ == "__main__":
    # 娴嬭瘯妯″瀷
    from torch_geometric.data import Data
    
    # 鍒涘缓娴嬭瘯鏁版嵁
    # 鍥綠i: 5涓妭鐐?
    x_i = torch.randn(5, 6)  # 6缁寸壒寰?
    edge_index_i = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data_i = Data(x=x_i, edge_index=edge_index_i)
    
    # 鍥綠j: 4涓妭鐐?
    x_j = torch.randn(4, 6)  # 6缁寸壒寰?
    edge_index_j = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data_j = Data(x=x_j, edge_index=edge_index_j)
    
    # 鍒涘缓妯″瀷
    model = create_model()
    
    # 鍓嶅悜浼犳挱锛坔ist(S)灏嗗湪妯″瀷鍐呴儴鑷姩璁＄畻锛?
    output = model(data_i, data_j)
    print(f"杈撳嚭褰㈢姸: {output.shape}")  # 搴旇鏄?[5, 4]
    print(f"杈撳嚭: \n{output}")
