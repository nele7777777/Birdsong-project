% 训练随机森林 (通常设置 50-100 棵树)
rfModel = TreeBagger(100, XTrain, yTrain, 'Method', 'classification');

% 预测 (注意：TreeBagger 的预测返回的是 cell 字符串)
[yPred_rf_raw, ~] = predict(rfModel, XTest);
yPred_rf = categorical(yPred_rf_raw);

% 计算准确率
accuracy_rf = sum(yPred_rf == yTest) / numel(yTest);
fprintf('Random Forest Baseline Accuracy: %.2f%%\n', accuracy_rf * 100);
%Random Forest Baseline Accuracy: 12.56%

figure;
confusionchart(yTest, yPred_svm, 'Title', 'SVM Baseline Confusion Matrix');