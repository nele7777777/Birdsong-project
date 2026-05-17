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
    print("Extracting features...")
    
    # Walk label subfolders produced by MATLAB export
    for label in os.listdir(base_path):
        folder_path = os.path.join(base_path, label)
        if not os.path.isdir(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Feature 1: duration (seconds)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    # Feature 2: spectral centroid (mean frequency height)
                    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    mean_freq = np.mean(centroid)
                    
                    data.append({
                        'filename': filename,
                        'original_label': label,
                        'duration': duration,
                        'mean_freq': mean_freq
                    })
                except Exception as e:
                    print(f"Failed to parse {filename}: {e}")
    return pd.DataFrame(data)

# --- 1. Data prep ---
# Set to your MATLAB syllable-clips output directory
CLIP_DIR = r'D:\Canary project\audio  May 2024\output_syllable_clips'
df = extract_features(CLIP_DIR)

# --- 2. Clustering ---
# Standardize so duration and frequency contribute equally
features = df[['duration', 'mean_freq']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

k = 24 
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# --- 3. Visualization ---
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['duration'], df['mean_freq'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster ID')

# Sparse text labels to avoid clutter
for i in range(0, len(df), len(df)//20):
    plt.text(df['duration'][i], df['mean_freq'][i], df['original_label'][i], fontsize=9)

plt.title(f'Birdsong Syllable Clustering (K={k})\nFeatures: Duration vs Mean Frequency')
plt.xlabel('Duration (seconds)')
plt.ylabel('Mean Frequency (Hz)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
