import os
import pandas as pd

def process_annotation_files(folder_path):
    '''删除 name 列中的 "_proposals"，并删除第四列值为 -1 的行'''
    total_files = 0
    modified_files_count = 0

    all_files = os.listdir(folder_path)
    csv_files = [f for f in all_files if f.lower().endswith('.csv')]

    print(f"📂 开始处理文件夹: {folder_path}")
    print(f"📊 检测到 CSV 文件总数: {len(csv_files)}\n" + "-"*50)

    for filename in csv_files:
        total_files += 1
        file_path = os.path.join(folder_path, filename)

        try:
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
            modified = False

            # 删除 name 列中的 "_proposals"
            if 'name' in df.columns:
                original_names = df['name'].astype(str).tolist()
                df['name'] = df['name'].astype(str).str.replace('_proposals', '', regex=False)
                updated_names = df['name'].tolist()

                if original_names != updated_names:
                    modified = True
                    print(f"✅ 标签修改 [{total_files}]: {filename}")
                    diff = set(original_names) - set(updated_names)
                    if diff:
                        print(f"   └─ 标签变更: {list(diff)} -> 已简化")
            else:
                print(f"⚠️ 跳过 [{total_files}]: {filename} (未找到 'name' 列)")

            # 删除第四列为 -1 的行（索引为 3）
            if df.shape[1] >= 4:
                col4_name = df.columns[3]
                before_rows = len(df)
                df = df[df[col4_name] != -1]
                after_rows = len(df)
                if before_rows != after_rows:
                    modified = True
                    print(f"🧹 删除行 [{total_files}]: {filename} - 移除 {before_rows - after_rows} 行 (第4列为 -1)")
            else:
                print(f"⚠️ 跳过 [{total_files}]: {filename} (列数不足 4 列)")
            
            # 删除第一列 name == "noise" 的行
            if 'name' in df.columns:
                before_rows_noise = len(df)
                df = df[df['name'].astype(str) != 'noise']
                after_rows_noise = len(df)

            if before_rows_noise != after_rows_noise:
                modified = True
                print(f"🗑️ 删除噪声行 [{total_files}]: {filename} - 移除 {before_rows_noise - after_rows_noise} 行 (name == 'noise')")


            # 如果有修改则保存
            if modified:
                df.to_csv(file_path, index=False, encoding='utf-8')
                modified_files_count += 1

        except Exception as e:
            print(f"❌ 处理 [{total_files}]: {filename} 时出错: {e}")

    print("-"*50)
    print(f"🏁 处理完成！")
    print(f"📈 总计扫描文件数: {total_files}")
    print(f"✨ 实际执行修改的文件数: {modified_files_count}")

if __name__ == "__main__":
    target_folder = r'D:\Aging bird project\1. Old-Young same individual\159\Y_annotated_excel'
    process_annotation_files(target_folder)
