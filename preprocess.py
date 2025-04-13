import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class YelpMultimodalDataset(Dataset):
    def __init__(self, data_path, text_tokenizer, image_transform, max_length=128):
        """
        初始化Yelp多模态数据集
        
        参数:
            data_path: Yelp数据集路径
            text_tokenizer: BERT tokenizer
            image_transform: 图像预处理转换
            max_length: 文本最大长度
        """
        self.data_path = data_path
        self.tokenizer = text_tokenizer
        self.transform = image_transform
        self.max_length = max_length
        
        # 加载业务数据
        with open(os.path.join(data_path, 'business.json'), 'r') as f:
            self.business_data = [json.loads(line) for line in f]
        
        # 转换为DataFrame以便于处理
        self.business_df = pd.DataFrame(self.business_data)
        
        # 加载评论数据
        with open(os.path.join(data_path, 'review.json'), 'r') as f:
            self.review_data = [json.loads(line) for line in f][:100000]  # 限制数量，便于处理
        
        self.review_df = pd.DataFrame(self.review_data)
        
        # 创建用户-商家-评价图
        self.users = self.review_df['user_id'].unique()
        self.businesses = self.business_df['business_id'].unique()
        
        # 用户和商家ID映射
        self.user_id_map = {uid: i for i, uid in enumerate(self.users)}
        self.business_id_map = {bid: i for i, bid in enumerate(self.businesses)}
        
        # 根据评分构建边
        self.edges = []
        self.ratings = []
        for _, row in self.review_df.iterrows():
            if row['user_id'] in self.user_id_map and row['business_id'] in self.business_id_map:
                user_idx = self.user_id_map[row['user_id']]
                business_idx = self.business_id_map[row['business_id']]
                self.edges.append((user_idx, business_idx + len(self.users)))
                self.ratings.append(row['stars'])
        
        # 图像路径
        self.photo_path = os.path.join(data_path, 'photos')
        
        # 每个商家关联的图片ID
        self.business_photos = {}
        for bid in self.businesses:
            photos = [f for f in os.listdir(self.photo_path) 
                     if f.startswith(bid) and f.endswith('.jpg')]
            self.business_photos[bid] = photos[:5]  # 每个商家最多取5张图
    
    def __len__(self):
        return len(self.review_df)
    
    def __getitem__(self, idx):
        review = self.review_df.iloc[idx]
        business_id = review['business_id']
        user_id = review['user_id']
        
        # 处理文本数据
        text = review['text']
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text_features = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        # 处理图像数据
        images = []
        if business_id in self.business_photos and self.business_photos[business_id]:
            for photo_id in self.business_photos[business_id]:
                try:
                    img_path = os.path.join(self.photo_path, photo_id)
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img)
                    images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading image {photo_id}: {e}")
        
        # 如果没有图像，使用零填充
        if not images:
            images.append(torch.zeros((3, 224, 224)))
        
        images = torch.stack(images, dim=0)
        
        # 商家元数据特征
        business_info = self.business_df[self.business_df['business_id'] == business_id].iloc[0]
        business_features = torch.tensor([
            business_info['stars'],  # 评分
            len(business_info['categories'].split(',')) if isinstance(business_info['categories'], str) else 0,  # 类别数量
            business_info['review_count'],  # 评论数
        ], dtype=torch.float)
        
        # 用户商家索引（用于图结构）
        user_idx = self.user_id_map.get(user_id, 0)
        business_idx = self.business_id_map.get(business_id, 0) + len(self.users)
        
        return {
            'text': text_features,
            'images': images,
            'business_features': business_features,
            'user_idx': user_idx,
            'business_idx': business_idx,
            'rating': torch.tensor(review['stars'], dtype=torch.float)
        }

def get_data_loaders(data_path, batch_size=32):
    # 初始化BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 图像转换
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建数据集
    dataset = YelpMultimodalDataset(data_path, tokenizer, image_transform)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 准备图结构数据
    edge_index = torch.tensor(dataset.edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(dataset.ratings, dtype=torch.float)
    
    return train_loader, test_loader, edge_index, edge_attr, len(dataset.users), len(dataset.businesses)