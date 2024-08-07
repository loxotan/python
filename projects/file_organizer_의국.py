import os
import shutil
import re

def copy_contents(src, dst):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src_item):
            copy_contents(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

def organize_folders(src_directory, dst_directory):
    folders_to_rename = []

    for year in os.listdir(src_directory):
        year_path = os.path.join(src_directory, year)
        if os.path.isdir(year_path) and re.match(r"\d{4}", year):
            for date_folder in os.listdir(year_path):
                if '-=' in date_folder or not date_folder.endswith('-'):
                    continue
                date_path = os.path.join(year_path, date_folder)
                if os.path.isdir(date_path):
                    for item_folder in os.listdir(date_path):
                        name, number, _ = item_folder.rsplit(' ', 2)
                        dst_folder_name = f"{name} {number}"
                        dst_folder_path = find_matching_folder(dst_directory, dst_folder_name)
                        src_item_path = os.path.join(date_path, item_folder)

                        if not dst_folder_path:
                            dst_folder_path = os.path.join(dst_directory, dst_folder_name)

                        dst_item_path = os.path.join(dst_folder_path, item_folder)
                        if not os.path.exists(dst_item_path):
                            copy_contents(src_item_path, dst_item_path)
                            print(f"Copied contents from {src_item_path} to {dst_item_path}")
                            folders_to_rename.append(date_path)

    for folder_path in set(folders_to_rename):
        try:
            new_folder_path = folder_path + '='
            os.rename(folder_path, new_folder_path)
            print(f"Renamed {folder_path} to {new_folder_path}")
        except PermissionError as e:
            print(f"Failed to rename {folder_path} due to permission error: {e}")

def find_matching_folder(directory, base_name):
    for folder_name in os.listdir(directory):
        if re.match(rf"^{re.escape(base_name)}", folder_name):
            return os.path.join(directory, folder_name)
    return None

src_directory = r'd:\IO photo\날짜별'
dst_directory = r'C:\Users\user\Desktop\환자별'
organize_folders(src_directory, dst_directory)