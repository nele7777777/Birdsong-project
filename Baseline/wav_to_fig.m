%% 1. 设置路径
inputRoot = 'D:\Aging bird project\1. Old-Young same individual\246\O_output_syllable_clips';    % 原始 wav 根目录
outputRoot = 'D:\Aging bird project\1. Old-Young same individual\246\O_syllables_fig';  % 生成图片的存放目录

% 创建输出根目录
if ~exist(outputRoot, 'dir'), mkdir(outputRoot); end

% 获取所有子文件夹（即标签名）
subFolders = dir(inputRoot);
subFolders = subFolders([subFolders.isdir] & ~startsWith({subFolders.name}, '.'));

%% 2. 遍历每个文件夹进行转换
for k = 1:length(subFolders)
    labelName = subFolders(k).name;
    fprintf('正在处理类别: %s...\n', labelName);
    
    % 创建对应的输出子文件夹
    labelOutputDir = fullfile(outputRoot, labelName);
    if ~exist(labelOutputDir, 'dir'), mkdir(labelOutputDir); end
    
    % 获取当前文件夹下所有 wav 文件
    wavFiles = dir(fullfile(inputRoot, labelName, '*.wav'));
    
    for i = 1:length(wavFiles)
        % 读取音频
        wavPath = fullfile(wavFiles(i).folder, wavFiles(i).name);
        [audio, fs] = audioread(wavPath);
        
        % --- 3. 计算梅尔语谱图 ---
        % 我们将窗口设为约 30ms，步长 10ms
        windowLen = round(0.03 * fs);
        overlapLen = round(0.02 * fs);
        
        % 计算 Mel Spectrogram (需要 Audio Toolbox)
        % 如果没有该工具箱，可以使用 pspectrum(audio, fs, 'spectrogram') 代替
        [S, f, t] = melSpectrogram(audio, fs, ...
            'Window', hamming(windowLen, 'periodic'), ...
            'OverlapLength', overlapLen, ...
            'NumBands', 128); % 128个梅尔频段，适合 CNN 输入
        
        % --- 4. 转化为图片并保存 ---
        % 将能量值转为分贝刻度，并归一化到 0-255 (灰度图或伪彩色)
        S_db = 10*log10(S + eps);
        
        % 调整大小以适应预训练网络 (如 GoogLeNet 需要 224x224)
        % 我们先用 imagesc 生成一个不带坐标轴的纯图
        fig = figure('Visible', 'off'); 
        imagesc(S_db);
        axis off; % 关掉坐标轴
        colormap(jet); % 使用彩色增强特征，CNN 很喜欢这种纹理
        
        % 设置图片大小并导出
        [~, fileName, ~] = fileparts(wavFiles(i).name);
        exportgraphics(gca, fullfile(labelOutputDir, [fileName, '.png']), 'Resolution', 72);
        
        close(fig); % 关闭隐藏的窗口，防止内存溢出
    end
end
fprintf('全部转换完成！\n');