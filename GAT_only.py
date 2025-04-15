import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATOnlyRecommender(nn.Module):
    def __init__(self, num_users, num_businesses, hidden_size=256, num_heads=4):
        super(GATOnlyRecommender, self).__init__()
        
        # 用户和商家嵌入
        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.business_embedding = nn.Embedding(num_businesses, hidden_size)
        
        # GAT层
        self.gat1 = GATConv(hidden_size, hidden_size, heads=num_heads)
        self.gat2 = GATConv(hidden_size * num_heads, hidden_size)
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, user_idx, business_idx, edge_index):
        # 获取用户和商家嵌入
        user_emb = self.user_embedding(user_idx)
        business_emb = self.business_embedding(business_idx - len(self.user_embedding.weight))
        
        # 准备所有节点的特征
        num_users = self.user_embedding.num_embeddings
        num_businesses = self.business_embedding.num_embeddings
        x = torch.zeros(num_users + num_businesses, user_emb.size(1)).to(user_emb.device)
        
        # 填充节点特征
        x[user_idx] = user_emb
        x[business_idx] = business_emb
        
        # 应用GAT层
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        
        # 获取当前批次的节点特征
        user_features = x[user_idx]
        business_features = x[business_idx]
        
        # 连接特征并预测评分
        concat_features = torch.cat([user_features, business_features], dim=1)
        rating_pred = self.predictor(concat_features)
        
        return rating_pred.squeeze()