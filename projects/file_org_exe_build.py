import os
import shutil
import re
from datetime import datetime
import json
import tkinter as tk
from tkinter import filedialog

def load_paths():
    try:
        with open('paths.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_paths(paths):
    with open('paths.json', 'w') as f:
        json.dump(paths, f)

def get_folder_path(prompt, initial_dir):
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=prompt, initialdir=initial_dir)
    root.destroy()
    return folder_path

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
        if not (os.path.isdir(year_path) and re.match(r"\d{4}$", year)):
            continue

        for date_folder in os.listdir(year_path):
            if '-=' in date_folder:
                continue
            date_path = os.path.join(year_path, date_folder)
            if not os.path.isdir(date_path):
                continue

            # 날짜 문자열 추출 (YYYY-MM-DD)
            try:
                date_str = datetime.strptime(date_folder[:10], '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError:
                # 날짜 파싱 실패 시 폴더명 전체를 사용
                date_str = date_folder

            for item_folder in os.listdir(date_path):
                orig_item_path = os.path.join(date_path, item_folder)
                if not os.path.isdir(orig_item_path):
                    raise ValueError(f"Error: '{item_folder}' is a file, not a folder. Please check the source directory.")

                # name, number, _ 로 나눌 수 있는지 시도
                parts = item_folder.rsplit(' ', 2)
                if len(parts) == 3:
                    name, number, _ = parts
                    new_folder_name = item_folder
                    src_item_path = orig_item_path
                else:
                    # 이름과 번호만 있는 경우
                    name, number = item_folder.rsplit(' ', 1)
                    # 새 폴더명: "name number YYYY-MM-DD"
                    new_folder_name = f"{name} {number} {date_str}"
                    new_src_item_path = os.path.join(date_path, new_folder_name)
                    os.rename(orig_item_path, new_src_item_path)
                    src_item_path = new_src_item_path

                # 환자별 폴더 (name + number)
                dst_folder_name = f"{name} {number}"
                dst_folder_path = os.path.join(dst_directory, dst_folder_name)

                # 최종 복사 경로: 환자별 폴더 안에 new_folder_name
                final_dst_path = os.path.join(dst_folder_path, new_folder_name)
                os.makedirs(final_dst_path, exist_ok=True)

                # 내부 파일/폴더 복사
                for entry in os.listdir(src_item_path):
                    s = os.path.join(src_item_path, entry)
                    d = os.path.join(final_dst_path, entry)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)



# 날짜별로 파일을 정리하는 함수
def organize_folders_by_date(source_dir, target_root, your_name):
    for year in os.listdir(source_dir):
        year_path = os.path.join(source_dir, year)
        if os.path.isdir(year_path) and re.match(r"\d{4}", year):  # 연도 형식 확인
            for date_folder in os.listdir(year_path):  # 상위 폴더만 탐색
                date_folder_path = os.path.join(year_path, date_folder)
                
                # '-'로 끝나지 않거나 '='로 끝나지 않는 폴더만 검사
                if not date_folder.endswith('-='):
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
                        
                        target_path = os.path.join(new_target_dir, your_name)
                        if not os.path.exists(target_path):
                            os.makedirs(target_path)  # your_name 폴더 생성
                            shutil.copytree(date_folder_path, target_path, dirs_exist_ok=True)

                    except ValueError as e:
                        print(f"Skipping {date_folder} due to ValueError: {e}")


# 폴더 이름에 '-=' 추가하는 함수
def rename_folders(src_directory):
    for year in os.listdir(src_directory):
        year_path = os.path.join(src_directory, year)
        if os.path.isdir(year_path) and re.match(r"\d{4}", year):
            for date_folder in os.listdir(year_path):
                date_path = os.path.join(year_path, date_folder)

                # Check if the folder does not end with '-='
                if not date_folder.endswith('-='):
                    # Create new folder name by adding '-=' to the folder name, not the path
                    new_folder_name = date_folder + '-='
                    new_folder_path = os.path.join(year_path, new_folder_name)

                    # Rename the folder
                    os.rename(date_path, new_folder_path)
                    print(f"Renamed {date_path} to {new_folder_path}")



def is_one_char_diff(s1, s2):
    """Returns True if s1 and s2 have exactly one character different."""
    if len(s1) != len(s2):
        return False
    diff_count = sum(1 for a, b in zip(s1, s2) if a != b)
    return diff_count == 1

def check_duplicates(file_set):
    """Check for duplicates in the given file set."""
    name_to_numbers = {}
    number_to_names = {}

    for file in file_set:
        if " " in file:
            name, number = file.rsplit(" ", 1)
            name_to_numbers.setdefault(name, []).append(number)
            number_to_names.setdefault(number, []).append(name)

    name_conflicts = []
    number_conflicts = []

    # Check for name conflicts
    for name, numbers in name_to_numbers.items():
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if is_one_char_diff(numbers[i], numbers[j]):
                    name_conflicts.append((f"{name} {numbers[i]}", f"{name} {numbers[j]}"))

    # Check for number conflicts
    for number, names in number_to_names.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if is_one_char_diff(names[i], names[j]):
                    number_conflicts.append((f"{names[i]} {number}", f"{names[j]} {number}"))

    return name_conflicts, number_conflicts

def combine_and_check_duplicates(folder1, folder2):
    """Combine contents of folder1 and folder2 using sets and check for duplicates."""
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Combine both sets
    combined_files = files1.union(files2)

    # Check for duplicates in the combined set
    name_conflicts, number_conflicts = check_duplicates(combined_files)

    return name_conflicts, number_conflicts


# 메인 함수
def main():
    try:
        # Load existing paths and name from the JSON file
        paths = load_paths()

        # Ask if the user wants to change the current paths or name
        change_paths_or_name = input("Do you want to change the current paths or name? (yes/no): ").strip().lower()

        if change_paths_or_name == 'yes':
            # Prompt for new paths if the user wants to change them
            src_directory = get_folder_path("원본 날짜별 폴더 선택", paths.get('src_directory', '/'))
            dst_directory = get_folder_path("임시 환자별 폴더 선택", paths.get('dst_directory', '/'))
            target_root = get_folder_path("서버 사진 폴더 선택", paths.get('target_root', '/'))
            original = get_folder_path("환자별로 정리된 원본 폴더 선택", paths.get('original', '/'))

            # Prompt for a new name if the user wants to change it
            your_name = input("당신의 이름을 입력하세요: ")

            # Update paths and name with new selections
            paths.update({
                'src_directory': src_directory,
                'dst_directory': dst_directory,
                'target_root': target_root,
                'original': original,
                'your_name': your_name
            })

            # Save the updated paths and name
            save_paths(paths)
            print("Paths and name have been saved successfully to paths.json.")
        
        else:
            # Use previously saved paths and name if the user doesn't want to change them
            print("Using previously saved paths and name.")
            your_name = paths.get('your_name', 'Default Name')  # Fallback to 'Default Name' if no name is found

        # Assign paths for further processing
        src_directory = paths['src_directory']
        dst_directory = paths['dst_directory']
        target_root = paths['target_root']
        original = paths['original']

        # Call the organizing and renaming functions
        organize_folders_by_patient(src_directory, dst_directory)
        organize_folders_by_date(src_directory, target_root, your_name)
        rename_folders(src_directory)

        # Check for conflicts and print them
        name_conflicts, number_conflicts = combine_and_check_duplicates(original, dst_directory)
        print("Name conflicts (same name, one digit different number):")
        for conflict in name_conflicts:
            print(conflict)
        print("\nNumber conflicts (same number, one character different name):")
        for conflict in number_conflicts:
            print(conflict)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()

