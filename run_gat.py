import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from GAT_only import GATOnlyRecommender

class ReviewDataset(Dataset):
    def __init__(self, reviews, user_mapping, business_mapping):
        self.user_idx = [user_mapping[review['user_id']] for review in reviews]
        self.business_idx = [business_mapping[review['business_id']] for review in reviews]
        self.ratings = [float(review['stars']) for review in reviews]
        
        self.user_idx = torch.LongTensor(self.user_idx)
        self.business_idx = torch.LongTensor(self.business_idx)
        self.ratings = torch.FloatTensor(self.ratings)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_idx': self.user_idx[idx],
            'business_idx': self.business_idx[idx],
            'rating': self.ratings[idx]
        }

def load_data(review_path, batch_size=32, test_size=0.2, top_users=10000, top_businesses=5000,       min_reviews=5, sample_ratio=None):
    # 加载JSON评论数据
    with open(review_path, 'r', encoding='utf-8') as f:
        reviews = [json.loads(line) for line in f]
    
    print(f"加载了 {len(reviews)} 条评论")

    user_counts = {}
    business_counts = {}
    
    for review in reviews:
        user_id = review['user_id']
        business_id = review['business_id']
        
        user_counts[user_id] = user_counts.get(user_id, 0) + 1
        business_counts[business_id] = business_counts.get(business_id, 0) + 1
    
    # 筛选至少有min_reviews条评论的用户和商家
    active_users = {user for user, count in user_counts.items() if count >= min_reviews}
    active_businesses = {business for business, count in business_counts.items() if count >=         min_reviews}
    
    # 获取最活跃的用户和商家
    top_users_set = set([user for user, _ in sorted(
        [(user, count) for user, count in user_counts.items() if user in active_users],
        key=lambda x: x[1], reverse=True)[:top_users]])
    
    top_businesses_set = set([business for business, _ in sorted(
        [(business, count) for business, count in business_counts.items() if business in        active_businesses],
        key=lambda x: x[1], reverse=True)[:top_businesses]])
    
    # 筛选同时包含活跃用户和活跃商家的评论
    reviews = [review for review in reviews 
                       if review['user_id'] in top_users_set 
                       and review['business_id'] in top_businesses_set]
    
    print(f"筛选后评论数量: {len(reviews)}")
    
    # 如果还需要进一步减少，可以随机采样
    if sample_ratio and sample_ratio < 1.0:
        import random
        random.seed(42)
        filtered_reviews = random.sample(filtered_reviews, int(len(filtered_reviews) * sample_ratio))
        print(f"采样后最终评论数量: {len(filtered_reviews)}")
    
    # 创建用户和商家的映射
    unique_users = set(review['user_id'] for review in reviews)
    unique_businesses = set(review['business_id'] for review in reviews)
    
    user_mapping = {user: idx for idx, user in enumerate(unique_users)}
    business_mapping = {business: idx + len(user_mapping) 
                        for idx, business in enumerate(unique_businesses)}
    
    print(f"总共 {len(user_mapping)} 个用户，{len(business_mapping)} 个商家")
    
    # 创建边索引 (user-business交互)
    edges = []
    for review in reviews:
        user_idx = user_mapping[review['user_id']]
        business_idx = business_mapping[review['business_id']]
        edges.append((user_idx, business_idx))
        edges.append((business_idx, user_idx))  # 双向边
    
    edge_index = torch.LongTensor(edges).t().contiguous()
    print(f"创建了 {edge_index.shape[1]} 条边")
    
    # 划分训练集和测试集
    train_reviews, test_reviews = train_test_split(reviews, test_size=test_size, random_state=42)
    
    # 创建数据集和数据加载器
    train_dataset = ReviewDataset(train_reviews, user_mapping, business_mapping)
    test_dataset = ReviewDataset(test_reviews, user_mapping, business_mapping)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, edge_index, len(user_mapping), len(business_mapping)

def train_epoch(model, train_loader, optimizer, criterion, edge_index, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="训练中"):
        # 将数据移至设备
        user_idx = batch['user_idx'].to(device)
        business_idx = batch['business_idx'].to(device)
        ratings = batch['rating'].to(device)
        
        # 前向传播
        predictions = model(user_idx, business_idx, edge_index)
        loss = criterion(predictions, ratings)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, edge_index, device, k_values=[5, 10, 20]):
    model.eval()
    
    # 收集所有预测结果
    all_user_predictions = {}
    all_user_ground_truth = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            user_idx = batch['user_idx'].to(device)
            business_idx = batch['business_idx'].to(device)
            ratings = batch['rating'].cpu().numpy()
            
            # 前向传播
            predictions = model(user_idx, business_idx, edge_index).cpu().numpy()
            
            # 记录预测和真实值
            for i in range(len(user_idx)):
                u_idx = user_idx[i].item()
                b_idx = business_idx[i].item()
                
                if u_idx not in all_user_predictions:
                    all_user_predictions[u_idx] = []
                    all_user_ground_truth[u_idx] = []
                
                all_user_predictions[u_idx].append((b_idx, predictions[i]))
                
                # 如果评分高于阈值，认为是相关项
                if ratings[i] >= 4.0:  # 可以调整此阈值
                    all_user_ground_truth[u_idx].append(b_idx)
    
    # 计算评估指标
    results = {}
    for k in k_values:
        precision_list, recall_list, ndcg_list, map_list = [], [], [], []
        
        for u_idx in all_user_predictions:
            # 对预测排序
            sorted_items = [item[0] for item in sorted(all_user_predictions[u_idx], 
                                                     key=lambda x: x[1], 
                                                     reverse=True)]
            relevant_items = all_user_ground_truth[u_idx]
            
            if len(relevant_items) == 0:
                continue
            
            # 计算前K个推荐的指标
            k_items = sorted_items[:k]
            relevant_and_recommended = set(relevant_items) & set(k_items)
            
            # Precision@K
            precision = len(relevant_and_recommended) / min(k, len(k_items)) if k_items else 0
            precision_list.append(precision)
            
            # Recall@K
            recall = len(relevant_and_recommended) / len(relevant_items) if relevant_items else 0
            recall_list.append(recall)
            
            # NDCG@K
            dcg = 0
            idcg = sum(1/np.log2(i+2) for i in range(min(k, len(relevant_items))))
            
            for i, item in enumerate(k_items):
                if item in relevant_items:
                    dcg += 1/np.log2(i+2)
            
            ndcg = dcg/idcg if idcg > 0 else 0
            ndcg_list.append(ndcg)
            
            # MAP@K (Mean Average Precision)
            hits = 0
            sum_precisions = 0
            
            for i, item in enumerate(k_items):
                if item in relevant_items:
                    hits += 1
                    sum_precisions += hits / (i + 1)
            
            ap = sum_precisions / min(len(relevant_items), k) if hits > 0 else 0
            map_list.append(ap)
        
        # 平均所有用户的指标
        results[f'Precision@{k}'] = np.mean(precision_list) if precision_list else 0
        results[f'Recall@{k}'] = np.mean(recall_list) if recall_list else 0
        results[f'NDCG@{k}'] = np.mean(ndcg_list) if ndcg_list else 0
        results[f'MAP@{k}'] = np.mean(map_list) if map_list else 0
    
    return results

def main():
    # 设置参数
    review_path = 'data/yelp/review.json'  # 修改为您的评论JSON文件路径
    batch_size = 64
    hidden_size = 64
    num_heads = 2
    epochs = 5
    lr = 0.001
    k_values = [5, 10, 20]
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, test_loader, edge_index, num_users, num_businesses = load_data(
        review_path, batch_size
    )
    edge_index = edge_index.to(device)
    
    # 初始化模型
    model = GATOnlyRecommender(
        num_users=num_users,
        num_businesses=num_businesses,
        hidden_size=hidden_size,
        num_heads=num_heads
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, edge_index, device)
        
        # 评估
        print(f"正在评估第 {epoch+1} 轮模型...")
        metrics = evaluate(model, test_loader, edge_index, device, k_values)
        
        # 打印结果
        print(f"轮次 {epoch+1}/{epochs}, 损失: {train_loss:.4f}")
        for k in k_values:
            print(f"  NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
            print(f"  Precision@{k}: {metrics[f'Precision@{k}']:.4f}")
            print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
            print(f"  MAP@{k}: {metrics[f'MAP@{k}']:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'gat_model.pt')
    print("训练完成，模型已保存。")

if __name__ == '__main__':
    main()