import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from data_preprocessing import get_data_loaders
from model import GATRecommender

class RecommendationEvaluator:
    """评估推荐系统性能的类，包含各种指标计算方法"""
    
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k=10):
        if len(recommended_items) == 0:
            return 0.0
        
        recommended_items = recommended_items[:k]
        relevant_and_recommended = set(relevant_items) & set(recommended_items)
        return len(relevant_and_recommended) / min(k, len(recommended_items))
    
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k=10):
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_items = recommended_items[:k]
        relevant_and_recommended = set(relevant_items) & set(recommended_items)
        return len(relevant_and_recommended) / len(relevant_items)
    
    @staticmethod
    def average_precision(recommended_items, relevant_items, k=10):
        if len(relevant_items) == 0:
            return 0.0
        
        relevant_items_set = set(relevant_items)
        recommended_items = recommended_items[:k]
        
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items_set:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        if hits == 0:
            return 0.0
            
        return sum_precisions / min(len(relevant_items), k)
    
    @staticmethod
    def mean_average_precision(recommended_items_list, relevant_items_list, k=10):
        if len(recommended_items_list) == 0:
            return 0.0
        
        aps = [
            RecommendationEvaluator.average_precision(recommended, relevant, k)
            for recommended, relevant in zip(recommended_items_list, relevant_items_list)
        ]
        
        return np.mean(aps)
    
    @staticmethod
    def ndcg_at_k(recommended_items, relevant_items, relevance_scores=None, k=10):
        if len(relevant_items) == 0:
            return 0.0
            
        relevant_items_set = set(relevant_items)
        
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant_items}
            
        recommended_items = recommended_items[:k]
        
        # 计算DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items):
            if item in relevant_items_set:
                rel = relevance_scores.get(item, 1.0)
                dcg += rel / np.log2(i + 2)
        
        # 计算理想DCG
        rel_scores = [relevance_scores.get(item, 1.0) for item in relevant_items]
        rel_scores.sort(reverse=True)
        rel_scores = rel_scores[:k]
        
        idcg = 0.0
        for i, rel in enumerate(rel_scores):
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    @staticmethod
    def hit_ratio_at_k(recommended_items, relevant_items, k=10):
        recommended_items = recommended_items[:k]
        relevant_items_set = set(relevant_items)
        
        for item in recommended_items:
            if item in relevant_items_set:
                return 1.0
        return 0.0
    
    @staticmethod
    def evaluate_all(recommended_items_list, relevant_items_list, k_values=[5, 10, 20]):
        results = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            hit_ratios = []
            
            for recommended, relevant in zip(recommended_items_list, relevant_items_list):
                precisions.append(RecommendationEvaluator.precision_at_k(recommended, relevant, k))
                recalls.append(RecommendationEvaluator.recall_at_k(recommended, relevant, k))
                ndcgs.append(RecommendationEvaluator.ndcg_at_k(recommended, relevant, None, k))
                hit_ratios.append(RecommendationEvaluator.hit_ratio_at_k(recommended, relevant, k))
            
            results[f'Precision@{k}'] = np.mean(precisions)
            results[f'Recall@{k}'] = np.mean(recalls)
            results[f'NDCG@{k}'] = np.mean(ndcgs)
            results[f'HR@{k}'] = np.mean(hit_ratios)
            results[f'MAP@{k}'] = RecommendationEvaluator.mean_average_precision(
                recommended_items_list, relevant_items_list, k)
        
        return results


def train_one_epoch(model, train_loader, optimizer, criterion, edge_index, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # 将所有数据移至GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 前向传播
        predictions = model(batch, edge_index)
        loss = criterion(predictions, batch['rating'])
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, edge_index, dataset, device, k_values=[5, 10, 20]):
    """评估模型性能"""
    model.eval()
    
    # 存储所有用户的推荐和真实相关项目
    all_user_recommendations = {}
    all_user_ground_truth = {}
    all_user_predicted_ratings = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 将所有数据移至GPU
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 获取用户和商家ID
            user_indices = batch['user_idx'].cpu().numpy()
            business_indices = batch['business_idx'].cpu().numpy()
            ratings = batch['rating'].cpu().numpy()
            
            # 前向传播
            predictions = model(batch, edge_index).cpu().numpy()
            
            # 存储每个用户的数据
            for i, user_idx in enumerate(user_indices):
                business_idx = business_indices[i]
                rating = ratings[i]
                predicted_rating = predictions[i]
                
                # 获取实际商家ID
                business_id = dataset.businesses[business_idx - len(dataset.users)]
                
                # 如果评分超过阈值，认为是相关的
                if rating >= 4.0:  # 可以调整此阈值
                    if user_idx not in all_user_ground_truth:
                        all_user_ground_truth[user_idx] = []
                    all_user_ground_truth[user_idx].append(business_id)
                
                # 存储预测评分，用于后续排序
                if user_idx not in all_user_predicted_ratings:
                    all_user_predicted_ratings[user_idx] = []
                all_user_predicted_ratings[user_idx].append((business_id, predicted_rating))
    
    # 为每个用户生成推荐列表（按预测评分排序）
    for user_idx, ratings in all_user_predicted_ratings.items():
        # 按预测评分降序排序
        sorted_businesses = [bid for bid, _ in sorted(ratings, key=lambda x: x[1], reverse=True)]
        all_user_recommendations[user_idx] = sorted_businesses
    
    # 准备评估数据
    recommended_items_list = []
    relevant_items_list = []
    
    for user_idx in all_user_ground_truth.keys():
        if user_idx in all_user_recommendations:
            recommended_items_list.append(all_user_recommendations[user_idx])
            relevant_items_list.append(all_user_ground_truth[user_idx])
    
    # 使用评估器计算所有指标
    evaluator = RecommendationEvaluator()
    results = evaluator.evaluate_all(recommended_items_list, relevant_items_list, k_values)
    
    return results


def plot_metrics(metrics_history, save_path):
    """绘制训练过程中的评估指标变化"""
    os.makedirs(save_path, exist_ok=True)
    
    # 提取epoch和指标数据
    epochs = []
    ndcg_values = []
    precision_values = []
    recall_values = []
    map_values = []
    
    for epoch, metrics in metrics_history.items():
        if metrics:  # 确保有数据
            epoch_num = int(epoch.split('_')[1])
            epochs.append(epoch_num)
            ndcg_values.append(metrics.get('NDCG@10', 0))
            precision_values.append(metrics.get('Precision@10', 0))
            recall_values.append(metrics.get('Recall@10', 0))
            map_values.append(metrics.get('MAP@10', 0))
    
    # 绘制指标随时间变化趋势
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, ndcg_values, 'o-', label='NDCG@10', linewidth=2)
    plt.plot(epochs, precision_values, 's-', label='Precision@10', linewidth=2)
    plt.plot(epochs, recall_values, '^-', label='Recall@10', linewidth=2)
    plt.plot(epochs, map_values, 'D-', label='MAP@10', linewidth=2)
    
    plt.title('Performance Metrics During Training', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'metrics_during_training.png'), dpi=300)
    plt.close()
    
    # 创建每个指标的单独图表
    metrics_data = {
        'NDCG@10': ndcg_values,
        'Precision@10': precision_values,
        'Recall@10': recall_values,
        'MAP@10': map_values
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (metric_name, values) in enumerate(metrics_data.items(), 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'{metric_name} During Training', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'individual_metrics.png'), dpi=300)
    plt.close()
    
    # 保存最终评估指标的条形图
    if epochs:
        final_metrics = metrics_history[f'epoch_{max(epochs)}']
        
        # 准备数据
        k_values = [5, 10, 20]  # 假设有这些K值
        metrics_types = ['NDCG', 'Precision', 'Recall', 'MAP', 'HR']
        
        data = []
        for k in k_values:
            for metric in metrics_types:
                metric_key = f'{metric}@{k}'
                if metric_key in final_metrics:
                    data.append({
                        'K': k,
                        'Metric': metric,
                        'Value': final_metrics[metric_key]
                    })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(15, 8))
        sns.barplot(x='K', y='Value', hue='Metric', data=df)
        plt.title('Final Evaluation Metrics', fontsize=16)
        plt.xlabel('K value', fontsize=14)
        plt.ylabel('Metric Value', fontsize=14)
        plt.legend(title='Metric', fontsize=12, title_fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_path, 'final_metrics.png'), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a multimodal recommendation system')
    parser.add_argument('--data_path', type=str, default='./data/yelp', help='Path to the Yelp dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in GAT')
    parser.add_argument('--save_path', type=str, default='./saved_models', help='Path to save the model')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20], 
                        help='K values for evaluation metrics (e.g., NDCG@K)')
    parser.add_argument('--results_path', type=str, default='./results', 
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, test_loader, edge_index, edge_attr, num_users, num_businesses = get_data_loaders(
        args.data_path, args.batch_size
    )
    edge_index = edge_index.to(device)
    
    # 初始化模型
    model = GATRecommender(
        num_users=num_users,
        num_businesses=num_businesses,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.results_path, exist_ok=True)
    
    # 训练循环
    best_ndcg = 0.0
    metrics_history = {}
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, edge_index, device)
        
        # 评估当前模型
        print(f"Evaluating model at epoch {epoch+1}...")
        metrics = evaluate_model(
            model, test_loader, edge_index, train_loader.dataset.dataset, device, args.k_values
        )
        
        # 保存指标历史
        metrics_history[f'epoch_{epoch+1}'] = metrics
        
        # 打印关键指标
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  NDCG@10: {metrics['NDCG@10']:.4f}")
        print(f"  Precision@10: {metrics['Precision@10']:.4f}")
        print(f"  Recall@10: {metrics['Recall@10']:.4f}")
        print(f"  MAP@10: {metrics['MAP@10']:.4f}")
        print(f"  Time: {elapsed_time:.2f}s")
        
        # 根据NDCG@10保存最佳模型
        if metrics['NDCG@10'] > best_ndcg:
            best_ndcg = metrics['NDCG@10']
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
            print(f"  New best model saved with NDCG@10: {best_ndcg:.4f}")
        
        # 保存当前模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'metrics': metrics,
            'best_ndcg': best_ndcg,
        }, os.path.join(args.save_path, f'model_epoch_{epoch+1}.pth'))
    
    # 保存所有评估指标历史
    with open(os.path.join(args.results_path, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=4)
    
    # 绘制评估指标变化图
    plot_metrics(metrics_history, args.results_path)
    
    print("\nTraining and evaluation completed!")
    print(f"Best NDCG@10: {best_ndcg:.4f}")
    print(f"Results saved to {args.results_path}")
    print(f"Models saved to {args.save_path}")

if __name__ == '__main__':
    main()