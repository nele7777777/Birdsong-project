% 使用多分类 SVM (一对多策略)
svmModel = fitcecoc(XTrain, yTrain);

% 预测
yPred_svm = predict(svmModel, XTest);

% 计算准确率
accuracy_svm = sum(yPred_svm == yTest) / numel(yTest);
fprintf('SVM Baseline Accuracy: %.2f%%\n', accuracy_svm * 100);
%SVM Baseline Accuracy: 13.49%