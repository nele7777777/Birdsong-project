function batchCutAudio_AutoMatch()
% -------------------------------------------------------------------------
% MATLAB 脚本: batchCutAudio_AutoMatch.m
% 修改内容：增加了对标签 'i' 的过滤逻辑，不再切割和保存该类音节。
% -------------------------------------------------------------------------

% --- 1. 配置路径 ---
ANNOTATION_DIR = 'D:\Aging bird project\1. Old-Young same individual\159\Y_annotated_excel';
AUDIO_DIR = 'D:\Aging bird project\1. Old-Young same individual\159\Y_denoised';
MAIN_OUTPUT_DIR = 'D:\Aging bird project\1. Old-Young same individual\159\Y_output_syllable_clips';

% 列名配置
START_TIME_COL = 'start_seconds';
END_TIME_COL = 'stop_seconds';
LABEL_COL = 'name';
CHANNEL_COL = 'channel';

% --- 2. 获取所有 CSV 文件列表 ---
csvFiles = dir(fullfile(ANNOTATION_DIR, '*.csv'));
if isempty(csvFiles)
    disp('错误: 在注释文件夹中未找到任何 .csv 文件。');
    return;
end
disp(['找到 ', num2str(length(csvFiles)), ' 个注释文件，开始匹配音频...']);

% --- 3. 循环处理每个注释文件 ---
for i = 1:length(csvFiles)
    csvName = csvFiles(i).name;
    csvPath = fullfile(ANNOTATION_DIR, csvName);
    
    audioBaseName = strrep(csvName, '_annotations.csv', '');
    audioName = [audioBaseName, '.wav'];
    audioPath = fullfile(AUDIO_DIR, audioName);
    
    if ~exist(audioPath, 'file')
        disp(['⚠️ 跳过: 找不到对应的音频文件 - ', audioName]);
        continue;
    end
    
    disp('---------------------------------------------------------');
    disp(['正在处理: ', audioName]);
    
    try
        T = readtable(csvPath);
        [y, fs] = audioread(audioPath);
    catch ME
        disp(['❌ 出错: 读取 ', csvName, ' 失败。', ME.message]);
        continue;
    end
    
    if ismember(CHANNEL_COL, T.Properties.VariableNames)
        T = T(T.(CHANNEL_COL) ~= -1, :);
    end
    
    total_rows = size(T, 1);
    if total_rows == 0
        disp('信息: 该注释文件没有有效音节（channel 均为 -1）。');
        continue;
    end

    % 计数器：记录实际导出的片段数
    exported_count = 0;

    % --- 4. 执行切割逻辑 ---
    for index = 1:total_rows
        label = char(T{index, LABEL_COL});
        
        % *********************************************************
        % 新增过滤逻辑：如果标签是 'i'，则直接跳过
        if strcmp(label, 'i')
            continue; 
        end
        % *********************************************************

        start_time_s = T{index, START_TIME_COL};
        end_time_s = T{index, END_TIME_COL};
        
        % 清理标签名以便建立文件夹
        label_clean = regexprep(label, '[ \/\\:*?"<>|]', '_'); 
        
        SAVE_DIR = fullfile(MAIN_OUTPUT_DIR, label_clean);
        if ~exist(SAVE_DIR, 'dir'), mkdir(SAVE_DIR); end
        
        % 索引计算
        start_sample = round(start_time_s * fs) + 1;
        end_sample = round(end_time_s * fs);
        
        % 边界检查
        if start_sample >= end_sample || start_sample < 1 || end_sample > length(y)
            continue;
        end
        
        % 切割与保存
        clip = y(start_sample:end_sample, :);
        [~, channel_name] = fileparts(AUDIO_DIR);
        output_filename = [channel_name, '_', audioBaseName, '_id', num2str(index, '%03d'), '_', label_clean, '.wav'];
        audiowrite(fullfile(SAVE_DIR, output_filename), clip, fs);
        
        exported_count = exported_count + 1;
    end
    
    disp(['✅ 完成: 已导出 ', num2str(exported_count), ' 个片段（已跳过所有标签为 "i" 的音节）。']);
end
disp('=========================================================');
disp('所有任务处理完毕！');
end