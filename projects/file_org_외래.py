import os
import shutil
import re
from datetime import datetime

# 요일을 한글로 변환하는 함수
def get_korean_weekday(weekday):
    korean_weekdays = ['월', '화', '수', '목', '금', '토', '일']
    return korean_weekdays[weekday]

# 파일 및 폴더를 복사하는 함수
def copy_contents(src, dst):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src_item):
            copy_contents(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

# 환자별로 파일을 분류하는 함수
def organize_folders_by_patient(src_directory, dst_directory):
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
                        dst_folder_path = os.path.join(dst_directory, dst_folder_name)
                        src_item_path = os.path.join(date_path, item_folder)

                        if not os.path.exists(dst_folder_path):
                            copy_contents(src_item_path, dst_folder_path)
                            print(f"Copied contents from {src_item_path} to {dst_folder_path}")

# 날짜별로 파일을 정리하는 함수
def organize_folders_by_date(source_dir, target_root):
    
    for year in os.listdir(source_dir):
        year_path = os.path.join(source_dir, year)
        if os.path.isdir(year_path) and re.match(r"\d{4}", year):  # 연도 형식 확인
            for date_folder in os.listdir(year_path):  # 상위 폴더만 탐색
                date_folder_path = os.path.join(year_path, date_folder)
                
                # '-'로 끝나지 않거나 '='로 끝나지 않는 폴더만 검사
                if not date_folder.endswith('-=') and date_folder.endswith('-'):
                    print(f"Checking directory: {date_folder}")
                    try:
                        # 날짜 파싱 시도
                        date = datetime.strptime(date_folder[:10], '%Y-%m-%d')
                        week_day = get_korean_weekday(date.weekday())
                        new_dir_name = date.strftime(f'%Y.%m.%d ({week_day})')
                        new_target_dir = os.path.join(target_root, f'{date.year}년도', f'{date.month}월', new_dir_name)
                        
                        # 대상 폴더가 없으면 생성
                        if not os.path.exists(new_target_dir):
                            os.makedirs(new_target_dir)
                        
                        target_path = os.path.join(new_target_dir, date_folder)
                        if not os.path.exists(target_path):
                            shutil.copytree(date_folder_path, target_path)

                    except ValueError as e:
                        print(f"Skipping {date_folder} due to ValueError: {e}")



# 폴더 이름에 '=' 추가하는 함수
def rename_folders(src_directory):
    for year in os.listdir(src_directory):
        year_path = os.path.join(src_directory, year)
        if os.path.isdir(year_path) and re.match(r"\d{4}", year):
            for date_folder in os.listdir(year_path):
                if '-=' in date_folder or not date_folder.endswith('-'):
                    continue
                date_path = os.path.join(year_path, date_folder)
                new_folder_path = date_path + '='
                os.rename(date_path, new_folder_path)
                print(f"Renamed {date_path} to {new_folder_path}")

# 메인 함수
def main():
    src_directory = r'E:\IO photo\날짜별'
    dst_directory = r'c:\Users\KNUDH\Desktop\환자별'
    target_root = r'c:\Users\KNUDH\Desktop\황날짜별'

    organize_folders_by_patient(src_directory, dst_directory)
    organize_folders_by_date(src_directory, target_root)
    rename_folders(src_directory)

if __name__ == "__main__":
    main()
