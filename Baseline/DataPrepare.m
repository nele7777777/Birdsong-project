% 假设 features 是你提取好的特征矩阵 [numFiles x numFeatures]
% 假设 labels 是你提取好的分类标签 {numFiles x 1} 的 cell 数组

% 1. 将标签转换为分类数组 (Categorical)
y = categorical(labels'); 

% 2. 特征标准化 (Standardization)
% 传统机器学习（尤其是 SVM）对特征缩放非常敏感
X = (features - mean(features)) ./ std(features);

% 3. 划分训练集与测试集 (80% 训练, 20% 测试)
cv = cvpartition(y, 'HoldOut', 0.2);
XTrain = X(training(cv), :);
yTrain = y(training(cv));
XTest = X(test(cv), :);
yTest = y(test(cv));