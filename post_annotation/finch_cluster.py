import os
import librosa
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings from very short signals
warnings.filterwarnings('ignore')

# --- Configuration ---
data_dirs = {
    "Old": r"D:\Aging bird project\1. Old-Young same individual\159\O_output_syllable_clips",
    "Young": r"D:\Aging bird project\1. Old-Young same individual\159\Y_output_syllable_clips"
}

def extract_sensitive_features(file_path):
    """Acoustic features: spectral entropy, flatness, and frequency bounds."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < sr * 0.01: return None  # drop very short clips
        
        n_fft = min(len(y), 2048)
        hop_length = n_fft // 4
        
        # 1. Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
        
        # 2. Spectral entropy (morphology stability)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        power_spec = np.sum(S**2, axis=0)
        power_spec /= (np.sum(power_spec) + 1e-10)
        spec_entropy = -np.sum(power_spec * np.log2(power_spec + 1e-10))

        # 3. Centroid and rolloff
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)
        
        # 4. RMS stability
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

# --- Collect features ---
all_features = []
season_labels = []   # Old / Young
syl_labels = []      # syllable type (subfolder name)

for season, path in data_dirs.items():
    print(f"Extracting features: {season}...")
    for subdir, _, files in os.walk(path):
        syllable = os.path.basename(subdir)
        for file in files:
            if file.lower().endswith('.wav'):
                feat = extract_sensitive_features(os.path.join(subdir, file))
                if feat:
                    all_features.append(feat)
                    season_labels.append(season)
                    syl_labels.append(syllable)

# --- UMAP ---
df = pd.DataFrame(
    all_features,
    columns=['Flatness', 'Entropy', 'Centroid', 'Rolloff', 'RMS_Std', 'Duration']
)

scaled_data = StandardScaler().fit_transform(df)

reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(scaled_data)

plot_df = pd.DataFrame({
    'UMAP1': embedding[:, 0],
    'UMAP2': embedding[:, 1],
    'Age': season_labels,
    'Syllable': syl_labels
})

# --- Plot ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=plot_df,
    x='UMAP1',
    y='UMAP2',
    hue='Age',
    style='Syllable',
    alpha=0.6,
    s=20
)
plt.title('UMAP of Syllables (Color=Age, Marker=Syllable)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
