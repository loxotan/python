import os
import shutil
import pandas as pd

def categorize(code):
    """
    Categorize based on the given code.
    """
    if code in ["FO", "GTR"]:
        return "1. FO, GTR"
    elif code == "peri-implantitis":
        return "2. peri-implantitis"
    elif code in ["GBR", "1st"]:
        return "3. GBR, impt"
    elif code in ["CL", "APF", "CTG", "FGG"]:
        return "4. CL, MGS"
    else:
        return "5. etc"

def open_folder(copying_folder):
    """
    Move contents from subfolders to the parent folder and remove empty subfolders.
    """
    for root, dirs, files in os.walk(copying_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                shutil.move(file_path, copying_folder)
            os.rmdir(subfolder_path)
        break  # Only process the top-level folder

def find_patient(unique_id, source_folder):
    """
    Find the folder containing the given unique ID in the source folder.
    """
    try:
        for folder_name in os.listdir(source_folder):
            if unique_id in folder_name:
                return os.path.join(source_folder, folder_name)
        return None
    except FileNotFoundError:
        return None

def organize(data, source_folder, directory_folder):
    """
    Organize folders based on the information in the data.
    """
    for index, row in data.iterrows():
        try:
            if isinstance(row.iloc[2], str):  # If C column is a name
                unique_id = str(row.iloc[3])  # Use D column as unique ID
                code = row.iloc[4]  # Use E column as code
            elif isinstance(row.iloc[2], (int, float)):  # If C column is a number
                unique_id = str(row.iloc[4])  # Use E column as unique ID
                code = row.iloc[5]  # Use F column as code
            else:
                continue

            copying_folder = find_patient(unique_id, source_folder)
            if copying_folder is None:
                print(f"{unique_id} 을 찾을 수 없습니다")
                continue

            target_folder_name = categorize(code)
            target_folder_path = os.path.join(directory_folder, target_folder_name)
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)

            # Skip if target already contains the folder
            if unique_id in os.listdir(target_folder_path):
                continue

            open_folder(copying_folder)

            # Rename the folder
            new_folder_name = f"{str(row.iloc[0])[:4]}_{row.iloc[1]}_{os.path.basename(copying_folder)}"
            new_folder_path = os.path.join(target_folder_path, new_folder_name)
            shutil.move(copying_folder, new_folder_path)

            print(f"{os.path.basename(copying_folder)} 폴더가 {target_folder_name} 폴더로 이동")
        except KeyError as e:
            print(f"Error processing row {index}: {e}")

# Paths
file_path = r"C:\Users\user\Desktop\술식별\count.xlsx"
source_folder = r"C:\Users\user\Desktop\환자별_241017_복사본"
directory_folder = r"C:\Users\user\Desktop\술식별"

# Load data
data = pd.read_excel(file_path)

# Run the organization process
organize(data, source_folder, directory_folder)
