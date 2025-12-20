import pandas as pd
from pydub import AudioSegment
import os

# --- 替换成你 ffmpeg.exe 文件的完整路径！ ---
# 示例：r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
AudioSegment.converter = r'C:\Users\lyuxuan\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe' 

# --- 1. 配置参数 ---
# 原始音频文件路径
INPUT_AUDIO_FILE = "D:\\Canary project\\audio  May 2024\\normalized loudness\\1_116_136_m_1.wav"  # !!! 更改为你的音频文件路径 !!!

# CSV 或 Excel 文件路径
ANNOTATION_FILE = "D:\\Canary project\\audio  May 2024\\train dataset\\Ch3_1_1\\1_116_136_m_1_annotations.csv"

# 切割后的音频保存目录
OUTPUT_DIR = "D:\\Canary project\\audio  May 2024\\output_syllable_clips"

# CSV/Excel 文件中，标记起始时间、结束时间和标签的列名
START_TIME_COL = 'start_seconds'  # 根据你的实际列名修改
END_TIME_COL = 'stop_seconds'      # 根据你的实际列名修改
LABEL_COL = 'name'      # 根据你的实际列名修改

# --- 2. 核心函数 ---

def batch_cut_audio_by_annotations(audio_path, annotation_path, output_dir, start_col, end_col, label_col):
    """
    根据 CSV/Excel 文件中的时间戳批量切割音频。
    """
    print(f"正在读取注释文件: {annotation_path}...")
    
    # 自动检测文件类型 (CSV 或 Excel)
    if annotation_path.lower().endswith('.csv'):
        df = pd.read_csv(annotation_path)
    elif annotation_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(annotation_path)
    else:
        print("错误: 注释文件格式不支持。请使用 .csv, .xlsx 或 .xls。")
        return

    # 检查所需的列是否存在
    required_cols = [start_col, end_col, label_col]
    if not all(col in df.columns for col in required_cols):
        print(f"错误: 注释文件中缺少以下列之一: {required_cols}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"正在加载音频文件: {audio_path}...")
    try:
        # Pydub 需要毫秒 (ms)，所以时间戳需要乘以 1000
        original_audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"加载音频文件失败。请检查文件路径和 FFmpeg 是否安装。错误: {e}")
        return

    total_clips = len(df)
    print(f"检测到 {total_clips} 个需要切割的音节片段。")
    
    # 遍历 DataFrame 中的每一行注释
    for index, row in df.iterrows():
        start_time_s = row[start_col]
        end_time_s = row[end_col]
        label = str(row[label_col]).replace(' ', '_').replace('/', '-') # 清理标签用于文件名

        # 转换为毫秒 (ms)
        start_ms = int(start_time_s * 1000)
        end_ms = int(end_time_s * 1000)

        # 确保时间戳有效
        if start_ms >= end_ms or start_ms < 0 or end_ms > len(original_audio):
            print(f"警告: 跳过第 {index+1} 行注释，时间戳无效或超出音频范围: {start_time_s}s - {end_time_s}s")
            continue

        # 切割音频
        clip = original_audio[start_ms:end_ms]

        # 构造输出文件名 (例如: clip_001_A.wav, clip_002_B.wav)
        output_filename = f"clip_{index+1:03d}_{label}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # 导出保存片段
        clip.export(output_path, format="wav")
        
        print(f"成功导出 ({index+1}/{total_clips}): {output_filename}")

    print("\n所有片段切割完成！")

# --- 3. 运行脚本 ---
if __name__ == "__main__":
    batch_cut_audio_by_annotations(
        INPUT_AUDIO_FILE,
        ANNOTATION_FILE,
        OUTPUT_DIR,
        START_TIME_COL,
        END_TIME_COL,
        LABEL_COL
    )
