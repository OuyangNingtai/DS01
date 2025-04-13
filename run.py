import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a multimodal recommendation system')
    parser.add_argument('--data_path', type=str, default='./data/yelp', 
                        help='Path to the Yelp dataset')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs')
    parser.add_argument('--hidden_size', type=int, default=256, 
                        help='Hidden size dimension')
    parser.add_argument('--num_heads', type=int, default=4, 
                        help='Number of attention heads in GAT')
    parser.add_argument('--save_path', type=str, default='./saved_models', 
                        help='Path to save the model')
    parser.add_argument('--results_path', type=str, default='./results', 
                        help='Path to save evaluation results')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20], 
                        help='K values for evaluation metrics (e.g., NDCG@K)')
    
    args = parser.parse_args()
    
    # 运行训练和评估
    cmd = f"python train_and_evaluate.py "\
         f"--data_path {args.data_path} "\
         f"--batch_size {args.batch_size} "\
         f"--lr {args.lr} "\
         f"--epochs {args.epochs} "\
         f"--hidden_size {args.hidden_size} "\
         f"--num_heads {args.num_heads} "\
         f"--save_path {args.save_path} "\
         f"--results_path {args.results_path} "\
         f"--k_values {' '.join(map(str, args.k_values))}"
    
    os.system(cmd)

if __name__ == "__main__":
    main()