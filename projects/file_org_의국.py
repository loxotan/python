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
    src_directory = r'E:\IO photo\날짜별'
    dst_directory = r'C:\Users\user\Desktop\환자별'
    target_root = r'C:\Users\user\Desktop\황날짜별'

    organize_folders_by_patient(src_directory, dst_directory)
    organize_folders_by_date(src_directory, target_root)
    rename_folders(src_directory)
    
    original = r'Y:\환자별'

    name_conflicts, number_conflicts = combine_and_check_duplicates(original, dst_directory)

    print("Name conflicts (same name, one digit different number):")
    for conflict in name_conflicts:
        print(conflict)

    print("\nNumber conflicts (same number, one character different name):")
    for conflict in number_conflicts:
        print(conflict)



if __name__ == "__main__":
    main()

