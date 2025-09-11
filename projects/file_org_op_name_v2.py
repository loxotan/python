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

def sync_folder_contents(src, dst):
    """Synchronizes contents from src to dst. Overwrites existing files."""
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                shutil.copytree(s, d)
            else:
                # If destination directory exists, sync its content recursively
                sync_folder_contents(s, d)
        else:
            shutil.copy2(s, d)

def update_existing_patients(source_folder, directory_folder, last_update_timestamp):
    """
    Step 1: Find already categorized patients and update their folders if the source has been modified.
    """
    print("--- 1단계: 기존 환자 폴더 업데이트 시작 ---")
    # Find all existing categorized patient folders
    categorized_folders = glob.glob(os.path.join(directory_folder, "*", "*_*_*"))
    
    updated_count = 0
    for dest_folder_path in categorized_folders:
        # Extract original patient folder name from the categorized name
        # e.g., from "2023_1_홍길동 1234567", extract "홍길동 1234567"
        match = re.search(r'\d+_\d+_(.*)$', os.path.basename(dest_folder_path))
        if not match:
            continue
        original_patient_name = match.group(1)
        
        source_patient_path = os.path.join(source_folder, original_patient_name)
        
        if not os.path.exists(source_patient_path):
            continue

        # Check if the source folder (or its contents) has been modified
        source_mtime = os.path.getmtime(source_patient_path)
        if source_mtime > last_update_timestamp:
            print(f"업데이트 발견: '{original_patient_name}' 폴더가 수정되었습니다. 내용물을 동기화합니다...")
            try:
                sync_folder_contents(source_patient_path, dest_folder_path)
                updated_count += 1
            except Exception as e:
                print(f"폴더 동기화 중 오류 발생 ({original_patient_name}): {e}")
    
    print(f"{updated_count}개의 기존 환자 폴더를 업데이트했습니다.")
    print("--- 1단계: 기존 환자 폴더 업데이트 완료 ---")

def add_new_patients(data, source_folder, directory_folder, last_update_timestamp):
    """
    Step 2: Find new patients from the Excel file (after last update) and add them.
    """
    print("\n--- 2단계: 신규 환자 추가 시작 ---")

    # Filter Excel data to only include entries after the last update timestamp
    try:
        # Ensure the date column is in datetime format for comparison
        date_column = data.columns[0]
        data[date_column] = pd.to_datetime(data[date_column])
        last_update_datetime = datetime.fromtimestamp(last_update_timestamp)
        
        new_patient_data = data[data[date_column] > last_update_datetime].copy()
        print(f"엑셀에서 마지막 업데이트 이후의 신규 환자 데이터 {len(new_patient_data)}건을 찾았습니다.")

        new_patient_data.iloc[:, 1] = new_patient_data.iloc[:, 1].astype(str)
        patient_data_list = new_patient_data.to_dict('records')
    except Exception as e:
        print(f"엑셀 데이터 처리 또는 필터링 중 오류: {e}")
        return

    added_count = 0
    for patient_info in patient_data_list:
        row_vals = [str(v) for v in patient_info.values()]
        
        # Extract patient ID
        id_candidates = [v for v in row_vals if re.fullmatch(r'\d{5,}', v)]
        if not id_candidates:
            continue
        unique_id = id_candidates[0]

        # Check if this patient is already organized
        existing = glob.glob(os.path.join(directory_folder, "*", f"*{unique_id}*"))
        if existing:
            continue # Skip if already exists

        # --- Found a new patient to add ---
        # Find the source patient folder
        source_patient_path = None
        for name in os.listdir(source_folder):
            if unique_id in name and os.path.isdir(os.path.join(source_folder, name)):
                source_patient_path = os.path.join(source_folder, name)
                break
        
        if not source_patient_path:
            print(f"신규 환자 '{unique_id}'의 폴더를 환자별 폴더(원본)에서 찾을 수 없습니다.")
            continue

        print(f"신규 환자 발견: {os.path.basename(source_patient_path)}")

        # Extract info for naming the new folder
        date_val = list(patient_info.values())[0]
        case_no_val = list(patient_info.values())[1] # Assuming case # is the 2nd column
        codes = {"FO","GTR","peri-implantitis","GBR", "CL","APF","CTG","FGG","1st","2nd","sinus"}
        code_candidates = [v for v in row_vals if v in codes]
        code = code_candidates[0] if code_candidates else "etc"

        year = pd.to_datetime(date_val).year
        case_no = str(case_no_val)
        
        # Create the new folder and copy contents
        target_dir = categorize(code)
        target_path = os.path.join(directory_folder, target_dir)
        os.makedirs(target_path, exist_ok=True)

        new_folder_name = f"{year}_{case_no}_{os.path.basename(source_patient_path)}"
        new_path = os.path.join(target_path, new_folder_name)

        try:
            shutil.copytree(source_patient_path, new_path)
            print(f"'{os.path.basename(source_patient_path)}' 폴더를 '{target_dir}' 카테고리로 새로 추가했습니다.")
            added_count += 1
        except Exception as e:
            print(f"폴더 복사 중 오류 발생 ({os.path.basename(source_patient_path)}): {e}")

    print(f"{added_count}명의 신규 환자를 추가했습니다.")
    print("--- 2단계: 신규 환자 추가 완료 ---")

def update_last_update_date(directory_folder):
    last_update_path = os.path.join(directory_folder, 'last_update.json')
    with open(last_update_path, 'w') as f:
        json.dump({'last_update_timestamp': datetime.now().timestamp()}, f)
    print(f"\nlast_update.json 파일이 현재 시간으로 업데이트되었습니다.")

if __name__ == '__main__':
    # --- 설정 ---
    source_folder = r"Y:\perio-photo\Photo (사진만.. 날짜 순대로 해주세요)\최수영 환자"
    directory_folder = r"C:\Users\최수영\Desktop\술식별"
    file_path = os.path.join(directory_folder, "count.xlsx")

    # --- 원본 폴더 확인 ---
    if not os.path.isdir(source_folder):
        print(f"오류: 원본 폴더 '{source_folder}'을(를) 찾을 수 없습니다.")
        exit()

    # --- 메인 로직 ---
    try:
        data = pd.read_excel(file_path, dtype=str)
    except FileNotFoundError:
        print(f"오류: 엑셀 파일 '{file_path}'을(를) 찾을 수 없습니다.")
        exit()

    last_update_file = os.path.join(directory_folder, 'last_update.json')
    
    if not os.path.exists(last_update_file):
        with open(last_update_file, 'w', encoding='utf-8') as f:
            initial_timestamp = 0.0
            json.dump({'last_update_timestamp': initial_timestamp}, f)
        print(f"'last_update.json' 파일이 생성되었습니다. 첫 실행으로 모든 폴더를 처리합니다.")

    with open(last_update_file, 'r', encoding='utf-8') as f:
        lu = json.load(f)
    last_update_timestamp = lu.get('last_update_timestamp', 0.0)

    # 1단계 실행
    update_existing_patients(source_folder, directory_folder, last_update_timestamp)
    
    # 2단계 실행
    add_new_patients(data, source_folder, directory_folder, last_update_timestamp)

    # 최종 시간 업데이트
    update_last_update_date(directory_folder)