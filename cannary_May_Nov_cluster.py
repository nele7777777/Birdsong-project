import os
import librosa
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略短信号带来的警告
warnings.filterwarnings('ignore')

# --- 配置区域 ---
data_dirs = {
    "Breeding Season": r"D:\Canary project\audio  May 2024\output_syllable_clips_May",
    "Non-breeding Season": r"D:\Canary project\audio Nov 2023\output_syllable_clips_Nov"
}

def extract_sensitive_features(file_path):
    """
    引入更敏感的声学指标：光谱熵、平坦度和频率边界
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < sr * 0.01: return None # 剔除极短噪声
        
        # 动态调整窗口大小
        n_fft = min(len(y), 2048)
        hop_length = n_fft // 4
        
        # 1. 光谱平坦度 (Spectral Flatness) - 核心敏感指标
        # 繁殖期音符偏向调性 (SF < -60dB)，非繁殖期偏向噪声 (SF -> 0dB) 
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
        
        # 2. 光谱熵 (Spectral Entropy) - 衡量形态稳定性
        # 非繁殖期的“塑性”会导致频谱更混乱，熵值更高 
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        power_spec = np.sum(S**2, axis=0)
        power_spec /= (np.sum(power_spec) + 1e-10) # 归一化
        spec_entropy = -np.sum(power_spec * np.log2(power_spec + 1e-10))

        # 3. 频率边界与质心
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
        # 光谱滚降 (Spectral Rolloff) - 区分频率范围分布 [cite: 198, 199]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)
        
        # 4. 能量稳定性 (RMS Variance)
        # 非繁殖期音节振幅波动通常更大 [cite: 120]
        rms = librosa.feature.rms(y=y, frame_length=n_fft)
        rms_std = np.std(rms)
        
        return [
            np.mean(flatness),
            spec_entropy,
            np.mean(centroid),
            np.mean(rolloff),
            rms_std,
            librosa.get_duration(y=y, sr=sr)
        ]
    except:
        return None

# --- 数据采集与处理 ---
all_features = []
labels = []

for season, path in data_dirs.items():
    print(f"提取特征中: {season}...")
    for subdir, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.wav'):
                feat = extract_sensitive_features(os.path.join(subdir, file))
                if feat:
                    all_features.append(feat)
                    labels.append(season)

# --- UMAP 降维与标准化 ---
df = pd.DataFrame(all_features, columns=['Flatness', 'Entropy', 'Centroid', 'Rolloff', 'RMS_Std', 'Duration'])

# 重要：UMAP 之前进行特征标准化，使所有指标权重均等
scaled_data = StandardScaler().fit_transform(df)

reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(scaled_data)

# --- 可视化 ---
plt.figure(figsize=(10, 7))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, alpha=0.6, s=20)
plt.title('UMAP with Sensitive Features (Entropy & Flatness)')
plt.show()