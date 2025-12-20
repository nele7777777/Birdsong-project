import os
import pandas as pd

def process_annotation_files(folder_path):
    # 初始化统计计数
    total_files = 0
    modified_files_count = 0
    
    # 获取文件夹内所有文件列表
    all_files = os.listdir(folder_path)
    csv_files = [f for f in all_files if f.lower().endswith('.csv')]
    
    print(f"📂 开始处理文件夹: {folder_path}")
    print(f"📊 检测到 CSV 文件总数: {len(csv_files)}\n" + "-"*50)

    for filename in csv_files:
        total_files += 1
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 使用针对 Excel 导出 CSV 优化的参数
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
            
            if 'name' in df.columns:
                # 记录原始数据用于比对
                original_names = df['name'].astype(str).tolist()
                
                # 执行替换逻辑
                df['name'] = df['name'].astype(str)
                df['name'] = df['name'].str.replace('_proposals', '', regex=False)
                
                # 检查是否真的发生了改变
                updated_names = df['name'].tolist()
                
                if original_names != updated_names:
                    # 只有发生改变时才保存并记录
                    df.to_csv(file_path, index=False, encoding='utf-8')
                    modified_files_count += 1
                    print(f"✅ 修改成功 [{total_files}]: {filename}")
                    # 找出具体哪个标签变了（可选，用于验证）
                    diff = set(original_names) - set(updated_names)
                    if diff:
                        print(f"   └─ 标签变更: {list(diff)} -> 已简化")
                else:
                    # 没有 _proposal 或 ' 的文件不输出详细信息，只统计
                    pass
            else:
                print(f"⚠️ 跳过 [{total_files}]: {filename} (未找到 'name' 列)")
        
        except Exception as e:
            print(f"❌ 处理 [{total_files}]: {filename} 时出错: {e}")

    print("-"*50)
    print(f"🏁 处理完成！")
    print(f"📈 总计扫描文件数: {total_files}")
    print(f"✨ 实际执行修改的文件数: {modified_files_count}")

if __name__ == "__main__":
    target_folder = r'D:\Canary project\audio  May 2024\train dataset\Train_Annotation_Excel'
    process_annotation_files(target_folder)