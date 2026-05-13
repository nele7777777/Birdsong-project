# filepath: c:\Users\lyuxuan\Project_Code\songsporperity_MultiChannel_finch.py
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import groupby
import librosa              # 新增：音频读取 + 基频估计
import numpy as np          # 新增：数值处理

# --- 配置区域 ---
channel_configs = [
    {"Age": "Old", "path": r"D:\Aging bird project\1. Old-Young same individual\159\O_annotated_excel"},
    {"Age": "Young", "path": r"D:\Aging bird project\1. Old-Young same individual\159\Y_annotated_excel"},
]

# 新增：每个 Age 的 syllable wav 根目录
AUDIO_BASE = {
    "Old": r"D:\Aging bird project\1. Old-Young same individual\159\O_output_syllable_clips",   # TODO: 改成你的 Old syllable wav 根目录
    "Young": r"D:\Aging bird project\1. Old-Young same individual\159\Y_output_syllable_clips" # TODO: 改成你的 Young syllable wav 根目录
}

output_excel_name = "Finch_syllable_Analysis_159.xlsx"

# 数据容器
song_summary_data = [] # 存储每首歌的统计指标
all_syllables = []
whole_song_patterns = []      # 现在改成存 Age + Pattern 的字典
same_type_intervals = []      # 新增：同类型 repeat 间隔
audio_syllable_f0 = []        # 新增：保存从 wav 算出来的 F0
all_adjacent_intervals = []   # 新增：所有相邻 syllable 间隔 (Age + name + next_name)

print("开始分通道处理数据...")

for config in channel_configs:
    ch_num = config["Age"]
    ch_path = config["path"]
    files = glob.glob(os.path.join(ch_path, "*.csv"))
    print(f"通道 {ch_num}: 找到 {len(files)} 个文件")

    for filename in files:
        try:
            base_name = os.path.basename(filename)
            df = pd.read_csv(filename)
            
            # 数据清洗 [cite: 471, 477]
            df['start_seconds'] = pd.to_numeric(df['start_seconds'], errors='coerce')
            df['stop_seconds'] = pd.to_numeric(df['stop_seconds'], errors='coerce')
            df = df.dropna(subset=['start_seconds', 'stop_seconds'])
            df = df[df['name'] != 'i'].sort_values('start_seconds')
            
            if df.empty: continue

            # 1. 计算歌曲基础指标 [cite: 125, 506]
            song_length = df['stop_seconds'].max() - df['start_seconds'].min()
            total_syllables = len(df)
            # 该首歌所有音符的平均时长 
            avg_duration = (df['stop_seconds'] - df['start_seconds']).mean()

            # 2. 生成结构序列 (用于 Pattern) [cite: 353, 501]
            sequence = df['name'].tolist()
            grouped_sequence = [(key, len(list(group))) for key, group in groupby(sequence)]
            # 详细 pattern 串，后面用于 Detailed_Patterns 表
            formatted_parts = [f"{name}({count})" for name, count in grouped_sequence]
            pattern_str = "-".join(formatted_parts)

            # 这里把 Age 和 FileName 一起存下来（每首歌一行）
            whole_song_patterns.append({
                "Age": ch_num,
                "FileName": base_name,
                "Pattern": pattern_str
            })

            # 3. 存储统计结果 
            song_info = {
                'Age': ch_num,
                'FileName': base_name,
                'Song_Length': round(song_length, 3),
                'Total_Syllable_Count': total_syllables,
                'Average_Syllable_Duration': round(avg_duration, 4),
            }
            song_summary_data.append(song_info)
            
            # 收集用于绘图/统计的数据
            df['duration'] = df['stop_seconds'] - df['start_seconds']
            tmp = df[['name', 'duration']].copy()
            tmp['Age'] = ch_num
            all_syllables.append(tmp)

            # === 新增：同类型 repeat interval 统计（Age + name） ===
            if len(df) > 1:
                df['next_start'] = df['start_seconds'].shift(-1)
                df['next_name'] = df['name'].shift(-1)
                df['interval'] = df['next_start'] - df['stop_seconds']
                df_valid_interval = df[df['interval'] >= 0].copy()

                # 1) 相同 syllable 的 repeat 间隔（你之前已有）
                repeats = df_valid_interval[df_valid_interval['name'] == df_valid_interval['next_name']].copy()
                if not repeats.empty:
                    tmp_int = repeats[['name', 'interval']].copy()
                    tmp_int['Age'] = ch_num
                    same_type_intervals.append(tmp_int)

                # 2) 所有相邻 syllable 的间隔 (name -> next_name)
                if not df_valid_interval.empty:
                    tmp_pairs = df_valid_interval[['name', 'next_name', 'interval']].copy()
                    tmp_pairs['Age'] = ch_num
                    all_adjacent_intervals.append(tmp_pairs)

        except Exception as e:
            print(f"处理通道 {ch_num} 文件 {base_name} 时出错: {e}")

print("开始计算每个音节 wav 的基频 F0 ...")
for age, root in AUDIO_BASE.items():
    if not os.path.isdir(root):
        print(f"警告: 音频根目录不存在, 跳过 {age}: {root}")
        continue

    # 每个子文件夹名 = syllable 名称
    for syll_name in os.listdir(root):
        syll_dir = os.path.join(root, syll_name)
        if not os.path.isdir(syll_dir):
            continue

        wav_files = glob.glob(os.path.join(syll_dir, "*.wav"))
        if not wav_files:
            continue

        for wav_path in wav_files:
            try:
                y, sr = librosa.load(wav_path, sr=None)
                if len(y) < sr * 0.01:  # < 10ms 的片段跳过
                    continue

                # 直接对整个片段估计基频序列
                f0_series = librosa.yin(
                    y,
                    fmin=80,    # 这些范围可按你物种音高再调整
                    fmax=8000,
                    sr=sr
                )
                f0_valid = f0_series[np.isfinite(f0_series)]
                if f0_valid.size == 0:
                    continue

                audio_syllable_f0.append({
                    "Age": age,
                    "name": syll_name,
                    "F0_Hz": float(f0_valid.mean())
                })
            except Exception as e:
                print(f"计算 F0 失败: {wav_path} -> {e}")

# --- 写入 Excel ---
print(f"正在写入结果至 {output_excel_name}...")
with pd.ExcelWriter(output_excel_name, engine='openpyxl') as writer:
    if song_summary_data:
        df_summary = pd.DataFrame(song_summary_data)
        df_summary = df_summary.sort_values(by=['Age', 'FileName'])

        # 追加：Old/Young 平均 Song_Length 以及差值
        avg_len = df_summary.groupby('Age')['Song_Length'].mean()
        old_avg = avg_len.get('Old', float('nan'))
        young_avg = avg_len.get('Young', float('nan'))
        diff_avg = young_avg - old_avg if pd.notna(old_avg) and pd.notna(young_avg) else float('nan')
        extra_rows = pd.DataFrame([
            {'Age': 'Old', 'FileName': 'Average', 'Song_Length': round(old_avg, 3)},
            {'Age': 'Young', 'FileName': 'Average', 'Song_Length': round(young_avg, 3)},
            {'Age': 'Diff(Young-Old)', 'FileName': 'Average_Song_Length_Diff', 'Song_Length': round(diff_avg, 3)}
        ])
        df_summary_out = pd.concat([df_summary, extra_rows], ignore_index=True)
        df_summary_out.to_excel(writer, sheet_name='Song_Summary', index=False)

    if all_syllables:
        big_df_syl = pd.concat(all_syllables, ignore_index=True)
        # 原来的 duration 描述统计
        stats_dur = big_df_syl.groupby(['Age', 'name'])['duration'].describe()

        # 新增：从 wav 计算的 F0, 按 Age + name 求平均并合并进来
        if audio_syllable_f0:
            df_f0 = pd.DataFrame(audio_syllable_f0)
            f0_stats = (
                df_f0
                .groupby(['Age', 'name'])['F0_Hz']
                .mean()
                .rename('F0_Mean_Hz')
            )
            stats_dur = stats_dur.join(f0_stats, how='left')

        # 统计 Young 相比 Old 的新增/缺失 syllable 类型数
        syl_sets = big_df_syl.groupby('Age')['name'].apply(set)
        old_set = syl_sets.get('Old', set())
        young_set = syl_sets.get('Young', set())

        added_in_young = sorted(list(young_set - old_set))
        missing_in_young = sorted(list(old_set - young_set))
        delta = len(young_set) - len(old_set)

        note_main = f"Compared to Old, Young has {delta:+d} unique syllable types (Young: {len(young_set)}, Old: {len(old_set)})."
        note_added = "Added in Young: " + (", ".join(added_in_young) if added_in_young else "None")
        note_missing = "Missing in Young: " + (", ".join(missing_in_young) if missing_in_young else "None")

        # 先写主表
        stats_dur.to_excel(writer, sheet_name='Syllable_Stats')
        current_row = len(stats_dur) + 2  # 空一行

        # === 新增：所有相邻 syllable 间隔 (Age + From + To) ===
        if all_adjacent_intervals:
            big_pairs = pd.concat(all_adjacent_intervals, ignore_index=True)
            pair_stats = (
                big_pairs
                .groupby(['Age', 'name', 'next_name'])['interval']
                .agg(Interval_Mean='mean', Interval_Count='size')
                .reset_index()
                .rename(columns={'name': 'From', 'next_name': 'To'})
                .sort_values(['Age', 'From', 'To'])
            )
            pair_stats.to_excel(
                writer,
                sheet_name='Syllable_Stats',
                index=False,
                startrow=current_row
            )
            current_row += len(pair_stats) + 2  # 再空一行给备注

        # 最后在下方追加说明
        pd.DataFrame({'Note': [note_main, note_added, note_missing]}).to_excel(
            writer,
            sheet_name='Syllable_Stats',
            index=False,
            startrow=current_row
        )

    # === 详细 pattern 统计表：每首歌一行，不做计数聚合 ===
    if whole_song_patterns:
        df_patterns = pd.DataFrame(whole_song_patterns)
        # 和第一个表一样，按 Age、FileName 排序
        df_patterns = df_patterns.sort_values(['Age', 'FileName'])
        df_patterns.to_excel(writer, sheet_name='Detailed_Patterns', index=False)

print("处理完成！")
