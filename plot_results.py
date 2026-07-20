import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_baselines(dataset_name):
    script_path = f"scripts/plot_{dataset_name}.py"
    if not os.path.exists(script_path):
        return {}
    
    with open(script_path, 'r') as f:
        code = f.read()
    
    plot_start = code.find("fig, ax = plt.subplots")
    if plot_start != -1:
        code = code[:plot_start]
        
    env = {}
    try:
        exec(code, env)
    except Exception as e:
        print(f"Error executing {script_path}: {e}")
        
    baselines = {}
    for k, v in env.items():
        if (k.endswith('_acc') or k.endswith('_auroc')) and not k.startswith('ec2_') and not k.startswith('rand_'):
            if isinstance(v, list):
                v = np.array([v, v, v])
            elif isinstance(v, np.ndarray):
                pass
            baselines[k] = np.array(v)
            
    return baselines

def plot_dataset_curves(dataset_name, seeds=[0, 1, 2], metric='acc'):
    """
    绘制指定数据集在不同预算(budget)下的性能曲线。
    :param metric: 'acc' 或者是 'auroc'
    """
    all_data = []
    
    # 读取该数据集所有 seed 的结果
    for seed in seeds:
        run_name = f"ec2_{dataset_name}_seed{seed}"
        csv_path = os.path.join("logs", run_name, "eval_results.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: Data file not found for {run_name} at {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        df['seed'] = seed
        all_data.append(df)
        
    if not all_data:
        print(f"No data found for dataset: {dataset_name}")
        return
        
    # 合并所有 seed 的数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 将所需指标重组为长格式(long format)方便 seaborn 绘图
    target_metric = f"ec2_{metric}"
    rand_metric = f"rand_{metric}"
    
    # 检查列是否存在（某些数据集如 cube_nm 可能没有 auroc 列）
    if target_metric not in combined_df.columns or rand_metric not in combined_df.columns:
        print(f"Metrics {target_metric} or {rand_metric} not found for dataset {dataset_name}. Skipping plot.")
        return
        
    melted_df = pd.melt(combined_df, 
                        id_vars=['budget', 'seed'], 
                        value_vars=[target_metric, rand_metric],
                        var_name='Strategy', 
                        value_name=metric.upper())
    
    # 替换策略名称使得图例更美观
    melted_df['Strategy'] = melted_df['Strategy'].map({
        target_metric: 'EC2 (Ours)',
        rand_metric: 'Random'
    })

    # 加载并合并 baselines
    baselines = load_baselines(dataset_name)
    baseline_records = []
    
    method_name_map = {
        'cae': 'CAE',
        'eddi': 'EDDI',
        'ga': 'Greedy-Abl',
        'iter': 'Iterative',
        'gdfs': 'GDFS',
        'aaco': 'AACO',
        'dime': 'DIME'
    }
    
    for name, data in baselines.items():
        base_method, base_metric = name.rsplit('_', 1)
        if base_metric != metric:
            continue
            
        if np.max(data) > 1.5:
            data = data / 100.0
            
        strategy_name = method_name_map.get(base_method.lower(), base_method.upper())
        
        for seed_idx, seed_data in enumerate(data):
            for budget, val in enumerate(seed_data):
                baseline_records.append({
                    'budget': budget,
                    'seed': seed_idx,
                    'Strategy': strategy_name,
                    metric.upper(): val
                })
                
    if baseline_records:
        baseline_df = pd.DataFrame(baseline_records)
        melted_df = pd.concat([melted_df, baseline_df], ignore_index=True)

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    # 绘制折线图，Seaborn 会自动计算多个 seed 的均值并用阴影画出置信区间(误差带)
    sns.lineplot(data=melted_df, x='budget', y=metric.upper(), 
                 hue='Strategy', style='Strategy', markers=True, dashes=False)
    
    plt.title(f'Performance on {dataset_name.capitalize()} dataset', fontsize=14, pad=15)
    plt.xlabel('Feature Acquisition Budget', fontsize=12)
    plt.ylabel(f'Test {metric.upper()}', fontsize=12)
    
    # 保存图像
    os.makedirs('figure', exist_ok=True)
    output_filename = os.path.join('figure', f'{dataset_name}_{metric}_curve.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved successfully to {output_filename}")

if __name__ == "__main__":
    # 您可以根据 run_all_seeds.sh 中的任务指定需要画图的数据集名称
    datasets = ['grid_stability', 'cube_nm', 'ckd', 'metabric', 'bank_marketing']
    
    for ds in datasets:
        # 画出 Accuracy 曲线
        plot_dataset_curves(ds, metric='acc')
        # 画出 AUROC 曲线
        plot_dataset_curves(ds, metric='auroc')
