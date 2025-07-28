import os
import shutil
import pandas as pd
import json
import re
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
    elif code in ["GBR", "impt", "1st", "2nd", "sinus"]:
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
    last_update_path = os.path.join(directory_folder, 'last_update.json')
    if not os.path.exists(last_update_path):
        print("last_update.json 파일이 없습니다.")
        return

    with open(last_update_path, 'r') as f:
        last_update_data = json.load(f)
        last_update_date = datetime.strptime(last_update_data['last_update'], '%Y-%m-%d')

    updated_folders = []
    for year_folder in os.listdir(date_organized_folder):
        if year_folder != last_update_date.strftime('%Y'):
            continue
        year_folder_path = os.path.join(date_organized_folder, year_folder)
        if not os.path.isdir(year_folder_path):
            continue
        for folder_name in os.listdir(year_folder_path):
            try:
                folder_date = datetime.strptime(folder_name[:10], '%Y-%m-%d')
                if folder_date > last_update_date:
                    updated_folders.append(os.path.join(year_folder_path, folder_name))
            except ValueError:
                continue

    for category_folder in os.listdir(directory_folder):
        category_path = os.path.join(directory_folder, category_folder)
        if not os.path.isdir(category_path):
            continue
        for patient_folder in os.listdir(category_path):
            patient_folder_path = os.path.join(category_path, patient_folder)
            unique_code = patient_folder.split(' ')[-1]
            for updated_folder in updated_folders:
                for sub in os.listdir(updated_folder):
                    if unique_code in sub:
                        source = os.path.join(updated_folder, sub)
                        for fname in os.listdir(source):
                            src = os.path.join(source, fname)
                            dst = os.path.join(patient_folder_path, fname)
                            if not os.path.exists(dst):
                                shutil.copy2(src, patient_folder_path)
                        print(f"{sub} 폴더가 업데이트되었습니다.")

def organize(data, source_folder, directory_folder, last_update_date):
    """
    Organize folders based on the information in the data.
    """
    for index, row in data.iterrows():
        if pd.to_datetime(row.iloc[0]) <= last_update_date:
            continue

        # 모든 값을 문자열로 변환
        row_vals = [str(x) for x in row.tolist()]

        # 1) 5자리 이상의 숫자 ID 추출
        id_candidates = [v for v in row_vals if re.fullmatch(r'\d{5,}', v)]
        if not id_candidates:
            print("ID를 찾을 수 없습니다:", row_vals)
            continue
        unique_id = id_candidates[0]

        # 2) 미리 정의된 코드 집합에서 코드 추출
        codes = {"FO","GTR","peri-implantitis","GBR", "CL","APF","CTG","FGG","1st","2nd","sinus"}
        code_candidates = [v for v in row_vals if v in codes]
        if not code_candidates:
            code_candidates.append("etc")
            continue
        code = code_candidates[0]

        copying_folder = find_patient(unique_id, source_folder)
        if copying_folder is None:
            print(f"{unique_id} 을 찾을 수 없습니다")
            continue

        # 연도 및 케이스 번호 추출
        year = pd.to_datetime(row.iloc[0]).year
        case_no = str(row.iloc[1])
        basename = os.path.basename(copying_folder)

        target_dir = categorize(code)
        target_path = os.path.join(directory_folder, target_dir)
        os.makedirs(target_path, exist_ok=True)

        new_folder = f"{year}_{case_no}_{basename}"
        new_path = os.path.join(target_path, new_folder)

        # 이미 정리된 경우 건너뜀
        existing = []
        for cat in ["1. FO, GTR","2. peri-implantitis","3. GBR, impt","4. CL, MGS","5. etc"]:
            existing += glob.glob(os.path.join(directory_folder, cat, f"*{unique_id}*"))
        if existing:
            continue

        shutil.copytree(copying_folder, new_path)
        open_folder(new_path)
        print(f"{basename} 폴더가 {target_dir} 폴더로 이동")

    print("정리 완료.")

def update_last_update_date(directory_folder):
    last_update_path = os.path.join(directory_folder, 'last_update.json')
    with open(last_update_path, 'w') as f:
        json.dump({'last_update': datetime.now().strftime('%Y-%m-%d')}, f)
    print("last_update.json 파일이 오늘 날짜로 업데이트되었습니다.")

if __name__ == '__main__':
    file_path = r"C:\Users\최수영\Desktop\술식별\count.xlsx"
    source_folder = r"Z:\환자별"
    directory_folder = r"C:\Users\최수영\Desktop\술식별"
    date_organized_folder = r"E:\IO photo\날짜별"

    data = pd.read_excel(file_path, dtype=str)
    last_update_file = os.path.join(directory_folder, 'last_update.json')
    with open(last_update_file, 'r') as f:
        lu = json.load(f)
    last_update_date = datetime.strptime(lu['last_update'], '%Y-%m-%d')

    updating_folder(date_organized_folder, directory_folder)
    organize(data, source_folder, directory_folder, last_update_date)
    update_last_update_date(directory_folder)
