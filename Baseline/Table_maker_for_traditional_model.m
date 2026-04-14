% 1. 设置音节文件夹路径
folderPath = 'D:\Aging bird project\1. Old-Young same individual\246\O_output_syllable_clips'; % 修改为你的文件夹路径
files = dir(fullfile(folderPath, '*', '*.wav')); 
numFiles = length(files);

% 初始化存储特征的矩阵和标签
features = zeros(numFiles, 4); 
labels = cell(numFiles, 1);

fprintf('开始提取特征，共检测到 %d 个文件...\n', numFiles);

for i = 1:numFiles
    % 读取音频
    [audio, fs] = audioread(fullfile(files(i).folder, files(i).name));
    
    % 特征 1: 持续时间 (Duration) - 衡量音节长短
    features(i, 1) = length(audio) / fs;
    
    % 特征 2: 平均频率 (Mean Frequency) - 衡量音调高低
    features(i, 2) = meanfreq(audio, fs);
    
    % 特征 3: 频谱质心 (Spectral Centroid) - 衡量音色（明亮度）
    features(i, 3) = mean(spectralCentroid(audio, fs));
    
    % 特征 4: 峰值频率 (Peak Frequency) - 能量最强的频率点
    [p, f] = periodogram(audio, [], [], fs);
    [~, maxIdx] = max(p);
    features(i, 4) = f(maxIdx);
    
    % 获取不带扩展名的文件名
    [~, nameOnly, ~] = fileparts(files(i).name);
    
    % 取最后一个字符作为标签
    labels{i} = nameOnly(end);
end

% 2. 转换为 Table 格式，方便导入 Classification Learner App
T = table(features(:,1), features(:,2), features(:,3), features(:,4), labels, ...
    'VariableNames', {'Duration', 'MeanFreq', 'Centroid', 'PeakFreq', 'Label'});

disp('特征提取完成！变量 T 已保存在工作区。');
disp('现在请在命令行输入 classificationLearner 开启 App。');