import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 步骤1: 更新文件路径
csv_file = r'D:/RT-DETR/results.csv'

# 步骤2: 读取CSV文件
df = pd.read_csv(csv_file)

# 确保'metrics/precision(B)'列存在
if 'metrics/precision(B)' in df.columns:
    # 步骤3: 绘制曲线图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    sns.lineplot(data=df, x=df.index, y='metrics/precision(B)', marker='o')  # 假设索引作为x轴

    plt.title('metrics/precision(B) Over Time')  # 图表标题
    plt.xlabel('Index')  # 假设索引作为x轴标签，根据实际情况调整
    plt.ylabel('Precision')  # y轴标签
    plt.xticks(rotation=45)  # 旋转x轴标签以便于阅读（如果适用）

    # 显示图表
    plt.tight_layout()
    plt.show()
else:
    print("Column 'metrics/precision(B)' not found in the DataFrame.")