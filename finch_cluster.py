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
    "Old": r"D:\Aging bird project\1. Old-Young same individual\159\O_output_syllable_clips",
    "Young": r"D:\Aging bird project\1. Old-Young same individual\159\Y_output_syllable_clips"
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
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
        
        # 2. 光谱熵 (Spectral Entropy) - 衡量形态稳定性
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        power_spec = np.sum(S**2, axis=0)
        power_spec /= (np.sum(power_spec) + 1e-10) # 归一化
        spec_entropy = -np.sum(power_spec * np.log2(power_spec + 1e-10))

        # 3. 频率边界与质心
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
        # 光谱滚降 (Spectral Rolloff) - 区分频率范围分布 [cite: 198, 199]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)
        
        # 4. 能量稳定性 (RMS Variance)
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
season_labels = []   # 原来的 Old / Young
syl_labels = []      # 新增：syllable 类型（子文件夹名）

for season, path in data_dirs.items():
    print(f"提取特征中: {season}...")
    for subdir, _, files in os.walk(path):
        syllable = os.path.basename(subdir)  # 子文件夹名 = syllable 类型
        for file in files:
            if file.lower().endswith('.wav'):
                feat = extract_sensitive_features(os.path.join(subdir, file))
                if feat:
                    all_features.append(feat)
                    season_labels.append(season)
                    syl_labels.append(syllable)

# --- UMAP 降维与标准化 ---
df = pd.DataFrame(
    all_features,
    columns=['Flatness', 'Entropy', 'Centroid', 'Rolloff', 'RMS_Std', 'Duration']
)

scaled_data = StandardScaler().fit_transform(df)

reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(scaled_data)

# 把标签也放进一个 DataFrame 方便画图
plot_df = pd.DataFrame({
    'UMAP1': embedding[:, 0],
    'UMAP2': embedding[:, 1],
    'Age': season_labels,
    'Syllable': syl_labels
})

# --- 可视化 ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=plot_df,
    x='UMAP1',
    y='UMAP2',
    hue='Age',          # 颜色：Old / Young
    style='Syllable',   # 形状：不同 syllable
    alpha=0.6,
    s=20
)
plt.title('UMAP of Syllables (Color=Age, Marker=Syllable)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()