import os
import pandas as pd

def process_annotation_files(folder_path):
    '''遍历文件夹中所有 CSV，批量重编码 name 列标签：
    - G → B
    '''
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

            # 统一重编码 name 列标签
            if 'name' in df.columns:
                original_names = df['name'].astype(str)

                # 映射规则：G→B
                rename_map = {
                    "G": "B",
                }

                new_names = original_names.replace(rename_map)
                changed_mask = original_names != new_names

                if changed_mask.any():
                    df['name'] = new_names
                    modified = True
                    changed_count = changed_mask.sum()
                    print(f"✅ 标签修改 [{total_files}]: {filename} - 共修改 {changed_count} 条标签 (G→B)")
            else:
                print(f"⚠️ 跳过 [{total_files}]: {filename} (未找到 'name' 列)")

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
    target_folder = r'D:\Canary project\audio Nov 2023\train dataset removeG\Ch6_annotated_excel'
    process_annotation_files(target_folder)
