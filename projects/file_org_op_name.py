import os
import shutil
import pandas as pd
from datetime import datetime

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

def updating_folder(source_folder, directory_folder):
    print("업데이트 시작...")
    """
    Update patient folders in the directory folder based on the latest changes in the source folder.
    """
    # Iterate through categories in the directory folder
    for category in sorted(os.listdir(directory_folder)):
        category_path = os.path.join(directory_folder, category)
        if not os.path.isdir(category_path):
            continue

        # Iterate through patient folders in the category
        for patient_folder in sorted(os.listdir(category_path)):
            patient_folder_path = os.path.join(category_path, patient_folder)
            if not os.path.isdir(patient_folder_path):
                continue

            # Extract unique ID from patient folder name (year_number_name unique ID)
            parts = patient_folder.split(' ')
            if len(parts) < 2:
                continue
            unique_id = parts[1]

            # Find the corresponding patient folder in the source folder
            target_patient = find_patient(unique_id, source_folder)
            if target_patient is None:
                continue

            # Get the last updated date from patient folder
            last_update_date = None
            for item in os.listdir(patient_folder_path):
                if len(item) >= 15 and item[-14] == ' ':
                    try:
                        item_date = datetime.strptime(item[-14:-4], '%Y-%m-%d')
                        if last_update_date is None or item_date > last_update_date:
                            last_update_date = item_date
                    except ValueError:
                        continue

            # Check and copy new updates from target_patient to patient_folder
            for subfolder in os.listdir(target_patient):
                subfolder_path = os.path.join(target_patient, subfolder)
                if os.path.isdir(subfolder_path):
                    folder_date_str = subfolder[-10:]
                    try:
                        folder_date = datetime.strptime(folder_date_str, '%Y-%m-%d')
                    except ValueError:
                        continue

                    if last_update_date is None or (last_update_date < folder_date):
                        for item in os.listdir(subfolder_path):
                            item_path = os.path.join(subfolder_path, item)
                            dest_path = os.path.join(patient_folder_path, item)
                            if os.path.isdir(item_path):
                                shutil.copytree(item_path, dest_path, dirs_exist_ok=True)
                            else:
                                shutil.copy2(item_path, dest_path)
            print(f"{patient_folder} 업데이트 완료")

    print("업데이트 완료! 새로운 술식을 확인합니다.")

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
            shutil.copytree(copying_folder, new_folder_path)

            print(f"{os.path.basename(copying_folder)} 폴더가 {target_folder_name} 폴더로 이동")
        except KeyError as e:
            print(f"Error processing row {index}: {e}")

# Paths
file_path = r"C:\Users\user\Desktop\술식별\count.xlsx"
source_folder = r"Y:\환자별"
directory_folder = r"C:\Users\user\Desktop\술식별"

# Load data
data = pd.read_excel(file_path)

# Run the updating process
updating_folder(source_folder, directory_folder)

# Run the organization process
organize(data, source_folder, directory_folder)
