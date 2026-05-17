import os
import pandas as pd

def process_annotation_files(folder_path):
    """Batch re-encode name column labels in all CSVs under folder_path.
    - C/E/G/X → B
    - R1 → R
    - JPF/W1 → P
    - L → B1
    - S → K
    - U → O
    - N1 → W
    - Y/Y' → C1
    """
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

            # Re-encode name column
            if 'name' in df.columns:
                original_names = df['name'].astype(str)

                # Mapping: C/E/G/X→B, R1→R, JPF/W1→P, etc.
                rename_map = {
                    "C": "B",
                    "C'": "B",
                    "E": "B",
                    "G": "B",
                    "G'": "B",
                    "X": "B",
                    "X'": "B",
                    "R1": "R",
                    "J": "P",
                    "F": "P",
                    "W1": "P",
                    "W1'": "P",
                    "L": "B1",
                    "S": "K",
                    "S'": "K",
                    "U": "O",
                    "U'": "O",
                    "N1": "W",
                    "Y": "C1",
                    "Y'": "C1",
                }

                new_names = original_names.replace(rename_map)
                changed_mask = original_names != new_names

                if changed_mask.any():
                    df['name'] = new_names
                    modified = True
                    changed_count = changed_mask.sum()
                    print(f"Labels updated [{total_files}]: {filename} - {changed_count} rows")
            else:
                print(f"Skip [{total_files}]: {filename} (no 'name' column)")

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
    target_folder = r'D:\Canary project\audio  May 2024\train dataset_compressed\Ch4_annotated_excel'
    process_annotation_files(target_folder)
