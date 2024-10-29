import os
import shutil
import pandas as pd
import json
from datetime import datetime
import glob

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

def updating_folder(date_organized_folder, directory_folder):
    print("업데이트 시작...")
    """
    Update patient folders in the directory folder based on the latest changes in the date_organized_folder.
    """
    # Load the last update date
    last_update_path = os.path.join(directory_folder, 'last_update.json')
    if not os.path.exists(last_update_path):
        print("last_update.json 파일이 없습니다.")
        return

    with open(last_update_path, 'r') as f:
        last_update_data = json.load(f)
        last_update_date = datetime.strptime(last_update_data['last_update'], '%Y-%m-%d')

    # Find updated folders
    updated_folders = []
    for year_folder in os.listdir(date_organized_folder):
        year_folder_path = os.path.join(date_organized_folder, year_folder)
        if year_folder != last_update_date.strftime('%Y'):
            continue

        if not os.path.isdir(year_folder_path):
            continue

        for folder_name in os.listdir(year_folder_path):
            try:
                folder_date_str = folder_name[:10]  # Extract the first 10 characters for date
                folder_date = datetime.strptime(folder_date_str, '%Y-%m-%d')
                if folder_date > last_update_date:
                    updated_folders.append(os.path.join(year_folder_path, folder_name))
            except ValueError:
                continue

    # Iterate over categorized folders
    for category_folder in os.listdir(directory_folder):
        category_path = os.path.join(directory_folder, category_folder)
        if not os.path.isdir(category_path):
            continue

        for patient_folder in os.listdir(category_path):
            patient_folder_path = os.path.join(category_path, patient_folder)
            unique_code = patient_folder.split(' ')[-1]

            # Check if updated folders contain the patient
            for updated_folder in updated_folders:
                for subfolder in os.listdir(updated_folder):
                    if unique_code in subfolder:
                        source_path = os.path.join(updated_folder, subfolder)

                        # Copy photos to the categorized patient folder
                        for file_name in os.listdir(source_path):
                            source_file_path = os.path.join(source_path, file_name)
                            target_file_path = os.path.join(patient_folder_path, file_name)
                            if not os.path.exists(target_file_path):
                                shutil.copy2(source_file_path, patient_folder_path)

                        print(f"{subfolder} 폴더가 업데이트되었습니다.")

def organize(data, source_folder, directory_folder, last_update_date):
    """
    Organize folders based on the information in the data.
    """
    for index, row in data.iterrows():
        # Skip rows that are not updated after the last update date
        if pd.to_datetime(row.iloc[0]) <= last_update_date:
            continue
        try:
            try:
                int(row.iloc[1])
            except ValueError:
                continue

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

            if copying_folder is None:
                print(f"{unique_id} 을 찾을 수 없습니다")
                continue

            # Define new_folder_path after verifying copying_folder exists
            new_folder_name = f"{str(row.iloc[0])[:4]}_{row.iloc[1]}_{os.path.basename(copying_folder)}"
            new_folder_path = os.path.join(target_folder_path, new_folder_name)

            # Skip if target already contains the folder
            # Check if a folder with the same unique ID already exists in the target folder
            existing_folders = glob.glob(os.path.join(target_folder_path, f'*{unique_id}*'))
            if existing_folders:
                continue

            open_folder(copying_folder)

            shutil.copytree(copying_folder, new_folder_path)

            print(f"{os.path.basename(copying_folder)} 폴더가 {target_folder_name} 폴더로 이동")
        except KeyError as e:
            print(f"Error processing row {index}: {e}")

def update_last_update_date(directory_folder):
    """
    Update the last update date in the last_update.json file to today.
    """
    last_update_path = os.path.join(directory_folder, 'last_update.json')
    new_update_data = {'last_update': datetime.now().strftime('%Y-%m-%d')}
    with open(last_update_path, 'w') as f:
        json.dump(new_update_data, f)
    print("last_update.json 파일이 오늘 날짜로 업데이트되었습니다.")

# Paths
file_path = r"C:\Users\user\Desktop\술식별\count.xlsx"
source_folder = r"Y:\환자별"
directory_folder = r"C:\Users\user\Desktop\술식별"
date_organized_folder = r"E:\IO photo\날짜별"

# Load data
data = pd.read_excel(file_path)

# Load last update date
last_update_path = os.path.join(directory_folder, 'last_update.json')
with open(last_update_path, 'r') as f:
    last_update_data = json.load(f)
last_update_date = datetime.strptime(last_update_data['last_update'], '%Y-%m-%d')

# Run the updating process
updating_folder(date_organized_folder, directory_folder)

# Run the organization process
organize(data, source_folder, directory_folder, last_update_date)

# Update the last update date
update_last_update_date(directory_folder)
