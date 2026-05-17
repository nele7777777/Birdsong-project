function batchCutAudio_AutoMatch()
% -------------------------------------------------------------------------
% MATLAB script: batchCutAudio_AutoMatch.m
% Skips syllables labeled 'i' (no cut/export for that class).
% -------------------------------------------------------------------------

% --- 1. Paths ---
ANNOTATION_DIR = 'D:\Aging bird project\offspring from old bird+tutor young father\Pair 4\tutee2568_annotated_excel';
AUDIO_DIR = 'D:\Aging bird project\offspring from old bird+tutor young father\Pair 4\tutee2568_denoised';
MAIN_OUTPUT_DIR = 'D:\Aging bird project\offspring from old bird+tutor young father\Pair 4\tutee2568_output_syllable_clips';

START_TIME_COL = 'start_seconds';
END_TIME_COL = 'stop_seconds';
LABEL_COL = 'name';
CHANNEL_COL = 'channel';

% --- 2. List CSV files ---
csvFiles = dir(fullfile(ANNOTATION_DIR, '*.csv'));
if isempty(csvFiles)
    disp('Error: no .csv files in annotation folder.');
    return;
end
disp(['Found ', num2str(length(csvFiles)), ' annotation files; matching audio...']);

% --- 3. Process each annotation file ---
for i = 1:length(csvFiles)
    csvName = csvFiles(i).name;
    csvPath = fullfile(ANNOTATION_DIR, csvName);
    
    audioBaseName = strrep(csvName, '_annotations.csv', '');
    audioName = [audioBaseName, '.wav'];
    audioPath = fullfile(AUDIO_DIR, audioName);
    
    if ~exist(audioPath, 'file')
        disp(['Skip: audio not found - ', audioName]);
        continue;
    end
    
    disp('---------------------------------------------------------');
    disp(['Processing: ', audioName]);
    
    try
        T = readtable(csvPath);
        [y, fs] = audioread(audioPath);
    catch ME
        disp(['Error reading ', csvName, ': ', ME.message]);
        continue;
    end
    
    if ismember(CHANNEL_COL, T.Properties.VariableNames)
        T = T(T.(CHANNEL_COL) ~= -1, :);
    end
    
    total_rows = size(T, 1);
    if total_rows == 0
        disp('Info: no valid syllables (all channel == -1).');
        continue;
    end

    exported_count = 0;

    % --- 4. Cut and export ---
    for index = 1:total_rows
        label = char(T{index, LABEL_COL});
        
        if strcmp(label, 'i')
            continue;
        end

        start_time_s = T{index, START_TIME_COL};
        end_time_s = T{index, END_TIME_COL};
        
        label_clean = regexprep(label, '[ \/\\:*?"<>|]', '_'); 
        
        SAVE_DIR = fullfile(MAIN_OUTPUT_DIR, label_clean);
        if ~exist(SAVE_DIR, 'dir'), mkdir(SAVE_DIR); end
        
        start_sample = round(start_time_s * fs) + 1;
        end_sample = round(end_time_s * fs);
        
        if start_sample >= end_sample || start_sample < 1 || end_sample > length(y)
            continue;
        end
        
        clip = y(start_sample:end_sample, :);
        [~, channel_name] = fileparts(AUDIO_DIR);
        output_filename = [channel_name, '_', audioBaseName, '_id', num2str(index, '%03d'), '_', label_clean, '.wav'];
        audiowrite(fullfile(SAVE_DIR, output_filename), clip, fs);
        
        exported_count = exported_count + 1;
    end
    
    disp(['Done: exported ', num2str(exported_count), ' clips (skipped label "i").']);
end
disp('=========================================================');
disp('All tasks finished.');
end
