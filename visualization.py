import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrices(json_path):
    # 1. 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = data['results']
    num_plots = len(results)
    
    # 2. 设置画布: 2行3列的网格排版，尺寸足够大以保证数字清晰
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 11))
    axes = axes.flatten() # 展平方便遍历
    
    # 统一设置全局颜色映射的最大值，这样不同图之间的颜色深浅就有可比性
    # 你的每个类别的样本数(support)大概在600-650之间
    VMAX = 650 

    # 3. 遍历每个保留率的结果并绘图
    for i, res in enumerate(results):
        rate = res['retention_rate']
        acc = res['classification']['accuracy']
        cm = np.array(res['confusion_matrix'])
        
        ax = axes[i]
        
        # 绘制热力图
        # annot=True 开启数字标签, fmt='d' 表示整数格式, cmap='Blues' 为蓝色渐变
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    cbar=(i == 4), # 只在最后一个图或者合适的位置显示色柱，避免画面太乱
                    vmax=VMAX, vmin=0, ax=ax, 
                    annot_kws={"size": 10}) # 调整格子内数字大小
        
        # 设置标题和坐标轴标签
        ax.set_title(f"Retention Rate: {rate}  |  Accuracy: {acc:.2%}", 
                     fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # 调整坐标轴刻度字体
        ax.tick_params(axis='both', which='major', labelsize=10)

    # 4. 处理多余的子图 (因为 2x3=6 个位置，只有 5 个图)
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j]) # 删除空白子图

    # 5. 全局标题与排版调整
    fig.suptitle('Confusion Matrix Evolution across Feature Retention Rates', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    # 给主标题留出空间
    plt.subplots_adjust(top=0.90) 
    
    # 6. 保存与展示
    output_filename = 'confusion_matrices_grid.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_filename}")
    
    # plt.show() # 如果在 Jupyter 或者有 GUI 的环境下可以取消注释

if __name__ == "__main__":
    # 替换为你的 json 文件实际路径
    file_path = "pretrain_eval_outputs/summary.json" 
    plot_confusion_matrices(file_path)