import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import groupby

# --- Configuration ---
channel_configs = [
    {"channel": 2, "path": r"D:\Canary project\audio Nov 2023\train dataset\Ch2_annotated_excel"},
    {"channel": 3, "path": r"D:\Canary project\audio Nov 2023\train dataset\Ch3_annotated_excel"},
    {"channel": 4, "path": r"D:\Canary project\audio Nov 2023\train dataset\Ch4_annotated_excel"},
    {"channel": 5, "path": r"D:\Canary project\audio Nov 2023\train dataset\Ch5_annotated_excel"},
    {"channel": 6, "path": r"D:\Canary project\audio Nov 2023\train dataset\Ch6_annotated_excel"}
]

output_excel_name = "Bird_Song_MultiChannel_Analysis.xlsx"
TARGET_PAIRS = [('R', 'H'), ('H', 'R1'), ('A', "H'")]

song_summary_data = []
all_syllables = []
whole_song_patterns = []

def compress_specific_pairs(rle_list, target_pairs):
    current_list = rle_list[:]
    new_list = []
    i = 0
    while i < len(current_list):
        if i == len(current_list) - 1:
            new_list.append(current_list[i])
            break
        curr_name, next_name = current_list[i][0], current_list[i+1][0]
        
        is_target = any(curr_name == p1 and next_name == p2 for p1, p2 in target_pairs)
        
        if is_target:
            combined_name = f"({curr_name}-{next_name})"
            new_list.append((combined_name, 1))
            i += 2
        else:
            new_list.append(current_list[i])
            i += 1
    return new_list

print("Processing channels...")

for config in channel_configs:
    ch_num = config["channel"]
    ch_path = config["path"]
    files = glob.glob(os.path.join(ch_path, "*.csv"))
    print(f"Channel {ch_num}: {len(files)} files")

    for filename in files:
        try:
            base_name = os.path.basename(filename)
            df = pd.read_csv(filename)
            
            df['start_seconds'] = pd.to_numeric(df['start_seconds'], errors='coerce')
            df['stop_seconds'] = pd.to_numeric(df['stop_seconds'], errors='coerce')
            df = df.dropna(subset=['start_seconds', 'stop_seconds'])
            df = df[df['name'] != 'i'].sort_values('start_seconds')
            
            if df.empty: continue

            song_length = df['stop_seconds'].max() - df['start_seconds'].min()
            total_syllables = len(df)
            unique_syllables = df['name'].nunique()
            avg_duration = (df['stop_seconds'] - df['start_seconds']).mean()

            sequence = df['name'].tolist()
            grouped_sequence = [(key, len(list(group))) for key, group in groupby(sequence)]
            compressed_seq = compress_specific_pairs(grouped_sequence, TARGET_PAIRS)
            formatted_pattern = "-".join([f"{name}({count})" for name, count in compressed_seq])

            song_info = {
                'Channel': ch_num,
                'FileName': base_name,
                'Song_Length': round(song_length, 3),
                'Total_Syllable_Count': total_syllables,
                'Unique_Syllable_Count': unique_syllables,
                'Average_Syllable_Duration': round(avg_duration, 4),
                'Structure_Pattern': formatted_pattern
            }
            song_summary_data.append(song_info)
            
            df['duration'] = df['stop_seconds'] - df['start_seconds']
            all_syllables.append(df[['name', 'duration']])

        except Exception as e:
            print(f"Error channel {ch_num} file {base_name}: {e}")

print(f"Writing {output_excel_name}...")
with pd.ExcelWriter(output_excel_name, engine='openpyxl') as writer:
    if song_summary_data:
        df_summary = pd.DataFrame(song_summary_data)
        df_summary = df_summary.sort_values(by=['Channel', 'FileName'])
        df_summary.to_excel(writer, sheet_name='Song_Summary', index=False)

    if all_syllables:
        big_df_syl = pd.concat(all_syllables)
        stats_dur = big_df_syl.groupby('name')['duration'].describe()
        stats_dur.to_excel(writer, sheet_name='Syllable_Stats')

print("Done.")
