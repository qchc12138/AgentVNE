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


class ModifiedNeuralTensorNetwork(nn.Module):
    """修改的神经张量网络"""
    def __init__(self, hidden_dim):
        super(ModifiedNeuralTensorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        # W: [hidden_dim, hidden_dim] 用于 Hi * W * Hj^T
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        # V1: [hidden_dim, 1] 用于 Hi * V1
        self.V1 = nn.Linear(hidden_dim, 1, bias=False)
        # V2: [hidden_dim, 1] 用于 V2 * Hj^T
        self.V2 = nn.Linear(hidden_dim, 1, bias=False)
        # 偏置项
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, Hi, Hj):
        """
        Hi: [N1, hidden_dim] - 图Gi的节点表示
        Hj: [N2, hidden_dim] - 图Gj的节点表示
        返回: K [N1, N2] - 相似度矩阵
        """
        N1, N2 = Hi.size(0), Hj.size(0)
        
        # 第一项: Hi * W * Hj^T
        # Hi @ W: [N1, hidden_dim] @ [hidden_dim, hidden_dim] = [N1, hidden_dim]
        # (Hi @ W) @ Hj^T: [N1, hidden_dim] @ [hidden_dim, N2] = [N1, N2]
        term1 = torch.matmul(torch.matmul(Hi, self.W), Hj.transpose(0, 1))
        
        # 第二项: Hi * V1 + V2 * Hj^T
        # Hi @ V1: [N1, hidden_dim] @ [hidden_dim, 1] = [N1, 1]
        Hi_proj = self.V1(Hi)  # [N1, 1]
        
        # Hj @ V2: [N2, hidden_dim] @ [hidden_dim, 1] = [N2, 1]
        Hj_proj = self.V2(Hj)  # [N2, 1]
        
        # 广播相加: [N1, 1] + [1, N2] = [N1, N2]
        term2 = Hi_proj + Hj_proj.transpose(0, 1)  # [N1, N2]
        
        # 最终输出
        #K = term1 + term2 + self.bias
        K = term1 + term2
        return K


class SimuVNE(nn.Module):
    """主要的SimuVNE神经网络模型"""
    def __init__(self, input_dim=6, hidden_dim=64, hist_dim=32):
        super(SimuVNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hist_dim = hist_dim  # 直方图的bins数量
        
        # GCN层 - 为图Gi单独的网络
        self.gcn1_i = GCNConv(input_dim, hidden_dim)
        self.gcn2_i = GCNConv(hidden_dim, hidden_dim)
        
        # GCN层 - 为图Gj单独的网络
        self.gcn1_j = GCNConv(input_dim, hidden_dim)
        self.gcn2_j = GCNConv(hidden_dim, hidden_dim)
        
        # Self-Attention层 - 为图Gi单独的注意力层
        self.self_attention_i = SelfAttention(hidden_dim)
        
        # Self-Attention层 - 为图Gj单独的注意力层
        self.self_attention_j = SelfAttention(hidden_dim)
        
        # 修改的Neural Tensor Network
        self.ntn = ModifiedNeuralTensorNetwork(hidden_dim)
        
        # 三层全连接网络

        # 加载配置
        # config = load_config(config_path)
        # config.update(kwargs)  # 命令行参数覆盖配置文件
        N2 = 10   # 后续仍需要修改，等cursor 搞定后吧
        self.fc1 = nn.Linear(N2, 256)  # K展平后的一行 + hist向量
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, N2)  # 输出单个值，表示该节点对的匹配分数
        
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
        
        # 提取直方图特征 hist(S)
        hist_S = self.calculate_histogram(S, bins=self.hist_dim)  # [hist_dim]
        
        # Self-Attention - 使用独立的注意力层
        Hi = self.self_attention_i(U_i)  # [N1, hidden_dim]
        Hj = self.self_attention_j(U_j)  # [N2, hidden_dim]
        
        # Neural Tensor Network
        K = self.ntn(Hi, Hj)  # [N1, N2]
        
        # 将K映射到[0, 1]
        K = torch.sigmoid(K)  # [N1, N2]
        
        # 相加得到组合分数
        output = K + S  # [N1, N2]
        
        # 对每一行做softmax（dim=1表示在N2维度上）
        output = F.softmax(output, dim=1)  # [N1, N2]
        
        # 通过三层全连接网络
        out = F.relu(self.fc1(output))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)  # [N1, N2]
        
        # 重塑为[N1, N2]
        output = F.softmax(out, dim=1)
        
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
