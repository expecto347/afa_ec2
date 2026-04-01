import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import sys

# ==========================================
# 1. 定义 Predictor 网络 (MLP 架构)
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
# 2. 掩码生成逻辑 (保持 Predictor 训练风格)
# ==========================================
def generate_mask(batch_size, d_in, device):
    # 为每个样本随机生成一个保留率 alpha ~ U(0, 1)
    alphas = torch.rand(batch_size, 1, device=device)
    # 生成均匀分布随机数，根据 alpha 生成 0/1 掩码
    probs = torch.rand(batch_size, d_in, device=device)
    mask = (probs < alphas).float()
    return mask

# ==========================================
# 3. 预训练主函数
# ==========================================
def pretrain_predictor():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('train_log.txt')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 检测是否输出到文件 (用于禁用 tqdm 的花哨功能)
    is_file_output = not sys.stdout.isatty()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 数据预处理：ToTensor + Flatten (28x28 -> 784)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 初始化模型 (参数与代码2完全一致)
    model = PredictorMLP(d_in=784, hidden=512, d_out=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 20 # Pretrain 通常需要较多轮次以覆盖各种稀疏度
    total_batches = len(train_loader)
    
    logger.info("=" * 50)
    logger.info("Starting Predictor Pretraining (MLP Version)...")
    logger.info(f"Total epochs: {epochs}, Batches per epoch: {total_batches}")
    logger.info("=" * 50)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # disable=True 让 tqdm 在日志模式下不显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=is_file_output)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # 1. 生成掩码
            mask = generate_mask(images.size(0), 784, device)
            
            # 2. 应用掩码
            masked_images = images * mask
            
            # 3. 前向传播 (输入拼接在模型内部完成)
            optimizer.zero_grad()
            outputs = model(masked_images, mask)
            loss = criterion(outputs, labels)
            
            # 4. 优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': avg_loss})
            
            # 定期输出日志 (每 100 个 batch)
            if batch_idx % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{total_batches}] Loss: {avg_loss:.6f}")
        
        epoch_loss = total_loss / total_batches
        logger.info(f"Epoch {epoch+1}/{epochs} completed - Average Loss: {epoch_loss:.6f}")

    # 保存权重，后续可加载到 CMIEstimator 中
    torch.save(model.state_dict(), "predictor_pretrain.pth")
    logger.info("=" * 50)
    logger.info("Pretraining complete. Model saved to predictor_pretrain.pth")
    logger.info("=" * 50)

if __name__ == "__main__":
    pretrain_predictor()