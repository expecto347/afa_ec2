import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 注: 2026-08 HPS 搜索得到的最佳组合 (mnist (20,4), miniboone (5,20), ...) 仅 seed 0 有结果；
# 为保持 3-seed 曲线配置一致，这里统一绘制默认配置的 ec2_<ds>_seed{0,1,2} 结果。

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
    
    # 读取该数据集所有 seed 的结果（默认配置 ec2_<ds>_seed<seed>）
    for seed in seeds:
        run_name = f"ec2_{dataset_name}_main_seed{seed}"
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
    ec2_label = 'Deep-EC2'
    melted_df['Strategy'] = melted_df['Strategy'].map({
        target_metric: ec2_label,
        rand_metric: 'Random'
    })

    # 最终配置 (按验证集选出, select_by_validation.py -> logs/ec2_<ds>_final_seed*)
    final_rows = []
    for seed in seeds:
        csv_final = os.path.join("logs", f"ec2_{dataset_name}_NOFINALOVERLAY_seed{seed}", "eval_results.csv")
        if os.path.exists(csv_final):
            df_f = pd.read_csv(csv_final)
            if target_metric in df_f.columns:
                final_rows.append(pd.DataFrame({'budget': df_f['budget'], 'seed': seed,
                                                'Strategy': 'Deep-EC2 (final)',
                                                metric.upper(): df_f[target_metric]}))
    if final_rows:
        melted_df = pd.concat([melted_df] + final_rows, ignore_index=True)

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
        'dime': 'DIME',
        'ol': 'OL (RL)'
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

    if dataset_name == 'bank_marketing':
        melted_df = melted_df[melted_df['budget'] < 10]
    elif dataset_name == 'cube_nm':
        melted_df = melted_df[melted_df['budget'] <= 55]

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    # 绘制折线图，Seaborn 会自动计算多个 seed 的均值并用阴影画出置信区间(误差带)
    order = ['Deep-EC2 (final)', ec2_label] + [x for x in melted_df['Strategy'].unique()
                                               if x not in ('Deep-EC2 (final)', ec2_label)]
    order = [x for x in order if x in set(melted_df['Strategy'])]
    ax = sns.lineplot(data=melted_df, x='budget', y=metric.upper(),
                      hue='Strategy', style='Strategy', markers=True, dashes=False,
                      hue_order=order, style_order=order)
    for line, name in zip(ax.get_lines(), order):
        if name.startswith('Deep-EC2'):
            line.set_linewidth(2.6)
    
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
    datasets = ['grid_stability', 'cube_nm', 'ckd', 'metabric', 'bank_marketing',
                'mnist', 'california_housing', 'miniboone',
                'syn1', 'syn2', 'syn3']

    for ds in datasets:
        # 所有数据集 EC2 与 baseline 均跑 seed 0/1/2，取均值±CI
        seeds = [0, 1, 2]
        # 画出 Accuracy 曲线
        plot_dataset_curves(ds, seeds=seeds, metric='acc')
        # 画出 AUROC 曲线
        plot_dataset_curves(ds, seeds=seeds, metric='auroc')
