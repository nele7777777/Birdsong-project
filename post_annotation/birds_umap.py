import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap

sns.set(style='white', context='notebook', rc={'figure.figsize': (8, 6)})

# Features exported from R
df = pd.read_csv(r'C:\Users\lyuxuan\workspace\Project_Code\159O.csv')

# Feature matrix: drop filename column (optionally drop 'selec' too)
X = df.drop(columns=['sound.files'])  # or ['sound.files', 'selec']

# Standardize (helps UMAP / clustering)
X_scaled = StandardScaler().fit_transform(X)

# UMAP embedding
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_scaled)  # (n_syllables, 2)

# Syllable label from filename (e.g. id001_b.wav -> b)
labels = df['sound.files'].str.extract(r'_([^_]+)\.wav$')[0]
labels_cat = labels.astype('category')
label_codes = labels_cat.cat.codes      # for scatter colors
label_names = labels_cat.cat.categories

sc = plt.scatter(embedding[:, 0], embedding[:, 1],
                 c=label_codes,
                 cmap='tab10', s=8, alpha=0.7)

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Syllable UMAP (159O)')

import matplotlib.patches as mpatches
handles = []
for code, name in enumerate(label_names):
    handles.append(mpatches.Patch(color=sc.cmap(sc.norm(code)), label=name))

plt.legend(handles=handles, title='Syllable type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
