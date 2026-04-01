import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import sys

# ==========================================
# 1. 定义 Predictor 网络 (与训练脚本相同)
# ==========================================
class PredictorMLP(nn.Module):
    def __init__(self, d_in=784, hidden=512, d_out=10, dropout=0.3):
        super(PredictorMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_in * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out)
        )

    def forward(self, x, mask):
        # x: [Batch, 784], mask: [Batch, 784]
        # 在特征维度进行拼接 -> [Batch, 1568]
        combined_input = torch.cat([x, mask], dim=1)
        return self.model(combined_input)

# ==========================================
# 2. 掩码生成逻辑
# ==========================================
def generate_mask(batch_size, d_in, device):
    # 为每个样本随机生成一个保留率 alpha ~ U(0, 1)
    alphas = torch.rand(batch_size, 1, device=device)
    # 生成均匀分布随机数，根据 alpha 生成 0/1 掩码
    probs = torch.rand(batch_size, d_in, device=device)
    mask = (probs < alphas).float()
    return mask

# ==========================================
# 3. 模型加载和评估
# ==========================================
def evaluate_model(model_path="predictor_pretrain.pth"):
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('eval_log.txt')
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ========== 加载模型 ==========
    logger.info("=" * 60)
    logger.info("Loading model...")
    logger.info("=" * 60)
    
    # 创建模型实例
    model = PredictorMLP(d_in=784, hidden=512, d_out=10).to(device)
    
    # 加载保存的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式 (禁用 dropout 等)
    
    logger.info(f"Model loaded successfully from {model_path}")
    logger.info(f"Model architecture:\n{model}")
    
    # ========== 准备测试数据 ==========
    logger.info("=" * 60)
    logger.info("Preparing test dataset...")
    logger.info("=" * 60)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # ========== 评估模型 ==========
    logger.info("=" * 60)
    logger.info("Evaluating model on test set...")
    logger.info("=" * 60)
    
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():  # 不需要计算梯度
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 生成掩码
            mask = generate_mask(images.size(0), 784, device)
            
            # 应用掩码
            masked_images = images * mask
            
            # 前向传播
            outputs = model(masked_images, mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 定期输出进度
            if (batch_idx + 1) % 20 == 0:
                current_acc = 100 * correct / total
                logger.info(f"Batch [{batch_idx+1}/{len(test_loader)}] "
                          f"Accuracy: {current_acc:.2f}% ({correct}/{total})")
    
    # ========== 最终结果 ==========
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total}")
    logger.info(f"Correct predictions: {correct}")
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Average Test Loss: {avg_loss:.6f}")
    logger.info("=" * 60)
    
    return accuracy, avg_loss

# ==========================================
# 4. 快速测试：不同掩码稀疏度下的准确率
# ==========================================
def evaluate_with_different_sparsity(model_path="predictor_pretrain.pth"):
    """测试模型在不同掩码稀疏度下的性能"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('eval_sparsity_log.txt')
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = PredictorMLP(d_in=784, hidden=512, d_out=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 准备测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    logger.info("=" * 60)
    logger.info("Testing model with different sparsity levels...")
    logger.info("=" * 60)
    
    sparsity_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for sparsity in sparsity_levels:
            correct = 0
            total = 0
            total_loss = 0
            
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 生成固定稀疏度的掩码
                if sparsity == 0.0:
                    # 完全掩码 (所有特征都被遮蔽)
                    mask = torch.zeros(images.size(0), 784, device=device)
                elif sparsity == 1.0:
                    # 完全分量 (没有特征被遮蔽)
                    mask = torch.ones(images.size(0), 784, device=device)
                else:
                    # 中间稀疏度
                    mask = torch.rand(images.size(0), 784, device=device)
                    mask = (mask < (1 - sparsity)).float()
                
                masked_images = images * mask
                outputs = model(masked_images, mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(test_loader)
            results[sparsity] = {'accuracy': accuracy, 'loss': avg_loss}
            
            logger.info(f"Sparsity: {sparsity:.0%} | Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.6f}")
    
    logger.info("=" * 60)
    return results

if __name__ == "__main__":
    # 方式1：评估带随机掩码的准确率
    logger = logging.getLogger(__name__)
    accuracy, loss = evaluate_model()
    
    print("\n")
    
    # 方式2：评估不同稀疏度下的准确率
    sparsity_results = evaluate_with_different_sparsity()
