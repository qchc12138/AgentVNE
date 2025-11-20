import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np



class SelfAttention(nn.Module):
    """自注意力机制层"""
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        """
        x: [N, hidden_dim] - 节点特征
        """
        Q = self.query(x)  # [N, hidden_dim]
        K = self.key(x)    # [N, hidden_dim]
        V = self.value(x)  # [N, hidden_dim]
        
        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [N, N]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # [N, hidden_dim]
        return output




class ColumnWiseTensorNetwork(nn.Module):
    """逐列神经张量网络，实现 hj * Wj * Hi^T"""
    def __init__(self, hidden_dim, num_nodes_j):
        super(ColumnWiseTensorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes_j = num_nodes_j
        self.W = nn.Parameter(torch.randn(num_nodes_j, hidden_dim, hidden_dim))
    
    def forward(self, Hi, Hj):
        """
        Hi: [N1, hidden_dim]
        Hj: [N2, hidden_dim]
        返回: Z [N2, N1]
        """
        N2 = Hj.size(0)
        if N2 != self.num_nodes_j:
            raise ValueError(
                f"N2={N2} 与设定的 num_nodes_j={self.num_nodes_j} 不一致，请确保目标图节点数量恒定。"
            )
        selected_W = self.W  # [N2, hidden_dim, hidden_dim]   以前怕不恒定，把这里做成了一个列表，挺有趣的一种写法
        hj_expanded = Hj.unsqueeze(1)  # [N2, 1, hidden_dim]
        hj_w = torch.matmul(hj_expanded, selected_W).squeeze(1)  # [N2, hidden_dim]
        Z = torch.matmul(hj_w, Hi.transpose(0, 1))  # [N2, N1]
        return Z


class SimuVNE(nn.Module):
    """主要的SimuVNE神经网络模型"""
    def __init__(self, input_dim=6, hidden_dim=64, hist_dim=32, num_nodes_j=10):
        super(SimuVNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hist_dim = hist_dim  # 直方图的bins数量
        self.num_nodes_j = num_nodes_j
        
        # GCN层 - 为图Gi单独的网络
        self.gcn1_i = GCNConv(input_dim, hidden_dim)
        self.gcn2_i = GCNConv(hidden_dim, hidden_dim)
        
        # GCN层 - 为图Gj单独的网络
        self.gcn1_j = GCNConv(input_dim, hidden_dim)
        self.gcn2_j = GCNConv(hidden_dim, hidden_dim)
        
        # Encoder层 - 为图Gi和Gj使用1层Transformer Encoder
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
        
        # 新的逐列NTN与encoder（使用PyTorch官方TransformerEncoderLayer，3层）
        self.ntn = ColumnWiseTensorNetwork(hidden_dim, num_nodes_j=num_nodes_j)
        encoder_layer_z = nn.TransformerEncoderLayer(
            d_model=num_nodes_j,
            nhead=1,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=False  # 使用 [seq_len, batch_size, d_model] 格式
        )
        self.encoder_z = nn.TransformerEncoder(encoder_layer_z, num_layers=3)
        
        self.dropout = nn.Dropout(0.1)
    
    def calculate_histogram(self, S, bins=32):
        """
        计算相似度矩阵S的直方图特征
        
        Args:
            S: [N1, N2] - 相似度矩阵
            bins: int - 直方图的分箱数量
        
        Returns:
            hist_features: [bins] - 直方图特征向量
        """
        # 将S展平为一维向量
        S_flat = S.view(-1)
        
        # 计算直方图
        # 使用torch.histc在[0, 1]范围内计算直方图
        hist = torch.histc(S_flat, bins=bins, min=0.0, max=1.0)
        
        # 归一化直方图
        hist = hist / (hist.sum() + 1e-8)
        
        return hist
        
    def forward(self, data_i, data_j):
        """
        data_i: 图Gi的数据 (x_i: [N1, 6], edge_index_i: [2, E1])
        data_j: 图Gj的数据 (x_j: [N2, 6], edge_index_j: [2, E2])
        返回: [N1, N2] - 每个节点对的匹配分数
        """
        x_i, edge_index_i = data_i.x, data_i.edge_index
        x_j, edge_index_j = data_j.x, data_j.edge_index
        
        # 要不要先过一个线性层呢？
        
        # 通过GCN处理图Gi - 使用独立的GCN网络，得到节点嵌入 U_i
        U_i = F.relu(self.gcn1_i(x_i, edge_index_i))
        U_i = self.dropout(U_i)
        U_i = F.relu(self.gcn2_i(U_i, edge_index_i))  # [N1, hidden_dim]
        
        # 通过GCN处理图Gj - 使用独立的GCN网络，得到节点嵌入 U_j
        U_j = F.relu(self.gcn1_j(x_j, edge_index_j))
        U_j = self.dropout(U_j)
        U_j = F.relu(self.gcn2_j(U_j, edge_index_j))  # [N2, hidden_dim]
        
        # 计算节点对相似度矩阵 S = σ(U_i * U_j^T)
        S = torch.sigmoid(torch.matmul(U_i, U_j.transpose(0, 1)))  # [N1, N2]
        
        # 提取直方图特征 hist(S) 并保留
        hist_S = self.calculate_histogram(S, bins=self.hist_dim)  # [hist_dim]
        self.last_histogram = hist_S
        

        # # Self-Attention - 使用独立的注意力层
        # Hi = self.self_attention_i(U_i)  # [N1, hidden_dim]
        # Hj = self.self_attention_j(U_j)  # [N2, hidden_dim]
        
        
        # 使用1层Transformer Encoder编码节点特征
        # U_i: [N1, hidden_dim] -> [N1, 1, hidden_dim] -> encoder -> [N1, hidden_dim]
        U_i_input = U_i.unsqueeze(1)  # [N1, hidden_dim] -> [N1, 1, hidden_dim]
        Hi_encoded = self.encoder_i(U_i_input)  # [N1, 1, hidden_dim]
        Hi = Hi_encoded.squeeze(1)  # [N1, 1, hidden_dim] -> [N1, hidden_dim]
        
        # U_j: [N2, hidden_dim] -> [N2, 1, hidden_dim] -> encoder -> [N2, hidden_dim]
        U_j_input = U_j.unsqueeze(1)  # [N2, hidden_dim] -> [N2, 1, hidden_dim]
        Hj_encoded = self.encoder_j(U_j_input)  # [N2, 1, hidden_dim]
        Hj = Hj_encoded.squeeze(1)  # [N2, 1, hidden_dim] -> [N2, hidden_dim]
        
        # 逐列NTN计算 Z (N2, N1)
        Z = self.ntn(Hi, Hj)  # [N2, N1]
        Z = Z.transpose(0, 1)  # [N1, N2]，此处 N2 恒定
        
        # 使用PyTorch官方TransformerEncoderLayer编码
        # TransformerEncoderLayer期望输入格式：[seq_len, batch_size, d_model] (batch_first=False)
        # Z是[N1, N2]，需要转换为[N1, 1, N2] = [seq_len, batch_size, d_model]
        
        # Z_input = Z.unsqueeze(1)  # [N1, N2] -> [N1, 1, N2]
        # Z_encoded = self.encoder_z(Z_input)  # [N1, 1, N2]
        # Z_prime = Z_encoded.squeeze(1)  # [N1, 1, N2] -> [N1, N2]
        # t = 1
        # output = F.softmax(Z_prime / t, dim=1)
        
        
        # # 将结果映射到0-1，然后按行归一化
        # Z_normalized = torch.sigmoid(Z)  # [N1, N2]，映射到0-1
        # output = F.softmax(Z_normalized, dim=1)  # [N1, N2]，按行归一化
        
        # 其他归一化方法（注释掉的替代方案）：
        # 方法1: L1归一化（按行除以行的L1范数，即行的绝对值之和）
        output = Z.abs() / (Z.abs().sum(dim=1, keepdim=True) + 1e-8)
        
        # 方法2: L2归一化（按行除以行的L2范数，即行的欧几里得范数）
        # output = F.normalize(Z_normalized, p=2, dim=1)
        
        # 方法3: Min-Max归一化（将每行映射到0-1范围，保持相对比例）
        # row_min = Z_normalized.min(dim=1, keepdim=True)[0]
        # row_max = Z_normalized.max(dim=1, keepdim=True)[0]
        # output = (Z_normalized - row_min) / (row_max - row_min + 1e-8)
        
        # 方法4: 简单的除以行和（类似softmax但不用exp，直接归一化）
        # output = Z_normalized / (Z_normalized.sum(dim=1, keepdim=True) + 1e-8)
        
        return output


def create_model(input_dim=6, hidden_dim=64, hist_dim=32):
    """创建模型实例"""
    model = SimuVNE(input_dim=input_dim, hidden_dim=hidden_dim, hist_dim=hist_dim)
    return model


if __name__ == "__main__":
    # 测试模型
    from torch_geometric.data import Data
    
    # 创建测试数据
    # 图Gi: 5个节点
    x_i = torch.randn(5, 6)  # 6维特征
    edge_index_i = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data_i = Data(x=x_i, edge_index=edge_index_i)
    
    # 图Gj: 4个节点
    x_j = torch.randn(4, 6)  # 6维特征
    edge_index_j = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data_j = Data(x=x_j, edge_index=edge_index_j)
    
    # 创建模型
    model = create_model()
    
    # 前向传播（hist(S)将在模型内部自动计算）
    output = model(data_i, data_j)
    print(f"输出形状: {output.shape}")  # 应该是 [5, 4]
    print(f"输出: \n{output}")
