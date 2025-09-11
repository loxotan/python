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

def updating_folder(date_organized_folder, directory_folder, organizer_name):
    print("업데이트 시작...")
    last_update_path = os.path.join(directory_folder, 'last_update.json')
    if not os.path.exists(last_update_path):
        print("last_update.json 파일이 없습니다.")
        return

    with open(last_update_path, 'r') as f:
        last_update_data = json.load(f)
        last_update_date = datetime.strptime(last_update_data['last_update'], '%Y-%m-%d').date()

    date_pattern = re.compile(r'.* (\d{4}-\d{2}-\d{2})\.(jpg|jpeg|png)$', re.IGNORECASE)
    print(f"마지막 업데이트 날짜 이후의 파일을 찾아 업데이트합니다: {last_update_date.strftime('%Y-%m-%d')}")

    # Construct a search path to specifically find the organizer's folders
    # Path: .../YYYY년도/MM월/YYYY.MM.DD (요일)/{organizer_name}
    organizer_search_path = os.path.join(date_organized_folder, "*", "*", "*", organizer_name)
    
    print(f"'{organizer_name}'의 폴더를 검색합니다...")
    for organizer_day_folder in glob.glob(organizer_search_path):
        if not os.path.isdir(organizer_day_folder):
            continue

        # Now walk through each found organizer folder
        for root, dirs, files in os.walk(organizer_day_folder):
            for filename in files:
                match = date_pattern.match(filename)
                if not match:
                    continue

                try:
                    file_date_str = match.group(1)
                    file_date = datetime.strptime(file_date_str, '%Y-%m-%d').date()

                    if file_date > last_update_date:
                        patient_folder_in_date_path = root
                        patient_folder_name = os.path.basename(patient_folder_in_date_path)
                        
                        unique_id_match = re.search(r'(\d{5,})$', patient_folder_name)
                        if not unique_id_match:
                            continue
                        
                        unique_id = unique_id_match.group(1)
                        
                        search_pattern = os.path.join(directory_folder, '*', f"*{unique_id}*")
                        dest_folders = glob.glob(search_pattern)

                        if not dest_folders:
                            continue

                        dest_folder_path = dest_folders[0]
                        
                        source_file_path = os.path.join(root, filename)
                        dest_file_path = os.path.join(dest_folder_path, filename)

                        if not os.path.exists(dest_file_path):
                            shutil.copy2(source_file_path, dest_file_path)
                            print(f"업데이트: '{os.path.basename(dest_folder_path)}' 폴더에 '{filename}' 추가됨")

                except (ValueError, IndexError):
                    continue

def organize(data, source_folder, directory_folder, last_update_date):
    """
    Organize folders based on the information in the data.
    """
    for index, row in data.iterrows():
        if pd.to_datetime(row.iloc[0]).date() <= last_update_date:
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
        print(f"{basename} 폴더가 {target_dir} 폴더로 이동")

    print("정리 완료.")

def update_last_update_date(directory_folder):
    last_update_path = os.path.join(directory_folder, 'last_update.json')
    with open(last_update_path, 'w') as f:
        json.dump({'last_update': datetime.now().strftime('%Y-%m-%d')}, f)
    print("last_update.json 파일이 오늘 날짜로 업데이트되었습니다.")

if __name__ == '__main__':
    # --- 설정 ---
    # photo_organizer_v2.0.exe가 있는 폴더 경로를 지정해주세요.
    # 예: r"C:\photo_app\dist\photo_organizer_v2.0"
    photo_organizer_path = r"C:\Users\최수영\Documents\python\python" # <--- 이 경로를 photo_organizer_v2.1.1.py가 있는 폴더로 수정했습니다.

    # 술식별 정리 결과를 저장할 폴더
    directory_folder = r"C:\Users\최수영\Desktop\술식별"
    # 술식 정보가 담긴 엑셀 파일
    file_path = os.path.join(directory_folder, "count.xlsx")

    # --- 경로 불러오기 ---
    paths_json_path = os.path.join(photo_organizer_path, 'paths.json')
    
    try:
        with open(paths_json_path, 'r', encoding='utf-8') as f:
            paths_data = json.load(f)
        
        source_folder = paths_data.get('dest_dir')
        date_organized_folder = paths_data.get('date_dest_dir')
        organizer_name = paths_data.get('organizer_name')

        if not all([source_folder, date_organized_folder, organizer_name]):
            print(f"오류: '{paths_json_path}'에 'dest_dir', 'date_dest_dir' 또는 'organizer_name' 경로가 없습니다.")
            exit()
            
    except FileNotFoundError:
        print(f"오류: 설정 파일 '{paths_json_path}'을(를) 찾을 수 없습니다.")
        print(f"'{photo_organizer_path}' 경로에 photo_organizer_v2.1.1.py와 paths.json이 있는지 확인해주세요.")
        exit()
    except json.JSONDecodeError:
        print(f"오류: '{paths_json_path}' 파일의 형식이 올바르지 않습니다.")
        exit()

    # --- 메인 로직 ---
    data = pd.read_excel(file_path, dtype=str)
    last_update_file = os.path.join(directory_folder, 'last_update.json')
    
    if not os.path.exists(last_update_file):
        oldest_date = pd.to_datetime(data.iloc[:, 0]).min()
        with open(last_update_file, 'w', encoding='utf-8') as f:
            initial_date = oldest_date - pd.Timedelta(days=1)
            json.dump({'last_update': initial_date.strftime('%Y-%m-%d')}, f)
        print(f"'last_update.json' 파일이 생성되었습니다. 초기 업데이트 날짜: {initial_date.strftime('%Y-%m-%d')}")

    with open(last_update_file, 'r', encoding='utf-8') as f:
        lu = json.load(f)
    last_update_date = datetime.strptime(lu['last_update'], '%Y-%m-%d').date()

    print("\n--- 폴더 업데이트 확인 시작 ---")
    updating_folder(date_organized_folder, directory_folder, organizer_name)
    print("--- 폴더 업데이트 확인 완료 ---\
")
    
    print("--- 신규 환자 정리 시작 ---")
    organize(data, source_folder, directory_folder, last_update_date)
    print("--- 신규 환자 정리 완료 ---\
")

    update_last_update_date(directory_folder)
