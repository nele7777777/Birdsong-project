#cluster
import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_features(base_path):
    data = []
    print("开始提取特征...")
    
    # 遍历 MATLAB 生成的按标签分类的文件夹
    for label in os.listdir(base_path):
        folder_path = os.path.join(base_path, label)
        if not os.path.isdir(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                try:
                    # 加载音频
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # 特征 1: 持续时间 (秒)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    # 特征 2: 频谱中心 (代表平均频率高度)
                    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    mean_freq = np.mean(centroid)
                    
                    data.append({
                        'filename': filename,
                        'original_label': label,
                        'duration': duration,
                        'mean_freq': mean_freq
                    })
                except Exception as e:
                    print(f"解析 {filename} 出错: {e}")
    return pd.DataFrame(data)

# --- 1. 数据准备 ---
# 替换为你 MATLAB 导出的输出目录
CLIP_DIR = r'D:\Canary project\audio  May 2024\output_syllable_clips'
df = extract_features(CLIP_DIR)

# --- 2. 聚类处理 ---
# 提取特征矩阵并标准化 (防止频率数值太大主导结果)
features = df[['duration', 'mean_freq']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 执行 K-means (假设你想分成 10 类，可以调整)
k = 24 
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# --- 3. 可视化 ---
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['duration'], df['mean_freq'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster ID')

# 标注一部分原始标签作为参考
for i in range(0, len(df), len(df)//20): # 每隔一段标注一个，防止太密集
    plt.text(df['duration'][i], df['mean_freq'][i], df['original_label'][i], fontsize=9)

plt.title(f'Birdsong Syllable Clustering (K={k})\nFeatures: Duration vs Mean Frequency')
plt.xlabel('Duration (seconds)')
plt.ylabel('Mean Frequency (Hz)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()