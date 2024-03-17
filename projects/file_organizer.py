import os
import shutil
import re

def copy_contents(src, dst):
    # 대상 폴더가 이미 존재하지 않으면 생성
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src_item):
            # 폴더인 경우 재귀적으로 처리
            copy_contents(src_item, dst_item)
        else:
            # 파일인 경우 복사
            shutil.copy2(src_item, dst_item)

def organize_folders(src_directory, dst_directory):
    for year in os.listdir(src_directory):
        year_path = os.path.join(src_directory, year)
        if os.path.isdir(year_path):
            for date_folder in os.listdir(year_path):
                date_match = re.match(r"(\d{4}-\d{2}-\d{2})", date_folder)
                if date_match:
                    date_path = os.path.join(year_path, date_folder)
                    if os.path.isdir(date_path):
                        for item_folder in os.listdir(date_path):
                            name, number, _ = item_folder.rsplit(' ', 2)
                            dst_folder_name = f"{name} {number}"
                            dst_folder_path = os.path.join(dst_directory, dst_folder_name)
                            src_item_path = os.path.join(date_path, item_folder)
                            dst_item_path = os.path.join(dst_folder_path, item_folder)

                            # 대상 폴더 내에 같은 이름의 소스 폴더가 이미 존재하는지 확인
                            if not os.path.exists(dst_item_path):
                                # 대상 폴더가 존재하지 않으면, 내용을 복사
                                copy_contents(src_item_path, dst_item_path)
                                print(f"Copied contents from {src_item_path} to {dst_item_path}")
                            else:
                                # 같은 이름의 폴더가 이미 존재하면, 건너뛰기
                                print(f"Folder {dst_item_path} already exists. Skipping...")

src_directory = 'c:/날짜별'
dst_directory = 'c:/환자별'
organize_folders(src_directory, dst_directory)
