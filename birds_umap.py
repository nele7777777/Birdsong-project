import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap

sns.set(style='white', context='notebook', rc={'figure.figsize': (8, 6)})

# 读取 R 生成的特征
df = pd.read_csv(r'C:\Users\lyuxuan\workspace\Project_Code\159O.csv')

# 特征矩阵：去掉文件名（需要的话也可以顺便去掉 selec）
X = df.drop(columns=['sound.files'])  # 或者 ['sound.files', 'selec']

# 标准化（一般对 UMAP/聚类有好处）
X_scaled = StandardScaler().fit_transform(X)

# UMAP 降维
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_scaled)  # (n_syllables, 2)

# 从文件名里抽取标签（举例：id001_b.wav 中的 b）
labels = df['sound.files'].str.extract(r'_([^_]+)\.wav$')[0]
# 把标签转成 category，拿到代码和名字
labels_cat = labels.astype('category')
label_codes = labels_cat.cat.codes      # 用来着色
label_names = labels_cat.cat.categories # 每个代码对应的名字

# 画图：用代码着色
sc = plt.scatter(embedding[:, 0], embedding[:, 1],
                 c=label_codes,
                 cmap='tab10', s=8, alpha=0.7)

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Syllable UMAP (159O)')

# 根据 label_names 构造图例
import matplotlib.patches as mpatches
handles = []
for code, name in enumerate(label_names):
    handles.append(mpatches.Patch(color=sc.cmap(sc.norm(code)), label=name))

plt.legend(handles=handles, title='Syllable type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()