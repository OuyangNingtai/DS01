import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel
from torch_geometric.nn import GATConv

class TextEncoder(nn.Module):
    def __init__(self, hidden_size=768):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, hidden_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS]标记的输出作为整个文本的表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_output)

class ImageEncoder(nn.Module):
    def __init__(self, hidden_size=768):
        super(ImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(768, hidden_size)
        
    def forward(self, pixel_values):
        batch_size, num_images, c, h, w = pixel_values.shape
        # 重塑以处理多张图像
        pixel_values = pixel_values.view(-1, c, h, w)
        outputs = self.vit(pixel_values=pixel_values)
        # 获取图像特征
        img_features = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记
        # 重塑回批次格式
        img_features = img_features.view(batch_size, num_images, -1)
        # 取平均，得到每个商家的图像表示
        img_features = torch.mean(img_features, dim=1)
        return self.fc(img_features)

class GATRecommender(nn.Module):
    def __init__(self, num_users, num_businesses, hidden_size=768, num_heads=8):
        super(GATRecommender, self).__init__()
        
        # 文本编码器和图像编码器
        self.text_encoder = TextEncoder(hidden_size)
        self.image_encoder = ImageEncoder(hidden_size)
        
        # 用户和商家嵌入
        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.business_embedding = nn.Embedding(num_businesses, hidden_size)
        
        # 商业特征处理
        self.business_feature_fc = nn.Linear(3, hidden_size)
        
        # GAT层
        self.gat1 = GATConv(hidden_size, hidden_size, heads=num_heads)
        self.gat2 = GATConv(hidden_size * num_heads, hidden_size)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def encode_nodes(self, text_features, images, business_features, user_idx, business_idx):
        # 文本编码
        text_embedding = self.text_encoder(text_features['input_ids'], text_features['attention_mask'])
        
        # 图像编码
        image_embedding = self.image_encoder(images)
        
        # 商业元数据编码
        business_meta_embedding = self.business_feature_fc(business_features)
        
        # 获取用户和商家基本嵌入
        user_emb = self.user_embedding(user_idx)
        business_emb = self.business_embedding(business_idx - user_emb.size(0))
        
        return text_embedding, image_embedding, business_meta_embedding, user_emb, business_emb
    
    def forward(self, data, edge_index):
        text_features = data['text']
        images = data['images']
        business_features = data['business_features']
        user_idx = data['user_idx']
        business_idx = data['business_idx']
        
        # 编码各个模态
        text_embedding, image_embedding, business_meta_embedding, user_emb, business_emb = self.encode_nodes(
            text_features, images, business_features, user_idx, business_idx
        )
        
        # 为所有节点准备特征
        num_users = self.user_embedding.num_embeddings
        num_businesses = self.business_embedding.num_embeddings
        x = torch.zeros(num_users + num_businesses, text_embedding.size(1)).to(text_embedding.device)
        
        # 更新节点特征
        x[user_idx] = user_emb
        x[business_idx] = (text_embedding + image_embedding + business_meta_embedding + business_emb) / 4
        
        # 应用GAT层
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        
        # 获取当前批次中的用户和商家节点特征
        user_features = x[user_idx]
        business_features = x[business_idx]
        
        # 融合特征以预测评分
        concat_features = torch.cat([
            user_features, 
            business_features, 
            text_embedding,
            image_embedding
        ], dim=1)
        
        rating_pred = self.fusion(concat_features)
        return rating_pred.squeeze()