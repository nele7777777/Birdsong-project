import os
import pandas as pd

def process_annotation_files(folder_path):
    """Strip '_proposals' from name column; drop rows with 4th column == -1 and name == 'noise'."""
    total_files = 0
    modified_files_count = 0

    all_files = os.listdir(folder_path)
    csv_files = [f for f in all_files if f.lower().endswith('.csv')]

    print(f"Processing folder: {folder_path}")
    print(f"CSV files found: {len(csv_files)}\n" + "-"*50)

    for filename in csv_files:
        total_files += 1
        file_path = os.path.join(folder_path, filename)

        try:
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
            modified = False

            # Remove "_proposals" from name column
            if 'name' in df.columns:
                original_names = df['name'].astype(str).tolist()
                df['name'] = df['name'].astype(str).str.replace('_proposals', '', regex=False)
                updated_names = df['name'].tolist()

                if original_names != updated_names:
                    modified = True
                    print(f"Labels updated [{total_files}]: {filename}")
                    diff = set(original_names) - set(updated_names)
                    if diff:
                        print(f"   changed: {list(diff)} -> simplified")
            else:
                print(f"Skip [{total_files}]: {filename} (no 'name' column)")

            # Drop rows where 4th column (index 3) is -1
            if df.shape[1] >= 4:
                col4_name = df.columns[3]
                before_rows = len(df)
                df = df[df[col4_name] != -1]
                after_rows = len(df)
                if before_rows != after_rows:
                    modified = True
                    print(f"Rows removed [{total_files}]: {filename} - {before_rows - after_rows} rows (4th column == -1)")
            else:
                print(f"Skip [{total_files}]: {filename} (fewer than 4 columns)")
            
            # Drop rows where name == "noise"
            if 'name' in df.columns:
                before_rows_noise = len(df)
                df = df[df['name'].astype(str) != 'noise']
                after_rows_noise = len(df)

            if before_rows_noise != after_rows_noise:
                modified = True
                print(f"Noise rows removed [{total_files}]: {filename} - {before_rows_noise - after_rows_noise} rows (name == 'noise')")


            if modified:
                df.to_csv(file_path, index=False, encoding='utf-8')
                modified_files_count += 1

        except Exception as e:
            print(f"Error [{total_files}]: {filename}: {e}")

    print("-"*50)
    print("Done.")
    print(f"Files scanned: {total_files}")
    print(f"Files modified: {modified_files_count}")

if __name__ == "__main__":
    target_folder = r'D:\Aging bird project\1. Old-Young same individual\159\Y_annotated_excel'
    process_annotation_files(target_folder)
