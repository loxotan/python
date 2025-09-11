import os
import re
import pandas as pd
from datetime import datetime

def count_followups_and_export_to_excel(main_directory):
    """
    Scans subdirectories in a given main directory to count unique dates in filenames,
    then exports the results to an Excel file.
    """
    print(f"지정한 폴더: '{main_directory}'의 분석을 시작합니다...")
    
    # Regex to find dates in YYYY-MM-DD or YYYY.MM.DD format
    date_pattern = re.compile(r'(\d{4}[-.]\d{2}[-.]\d{2})')
    
    results = []

    # Loop through each item in the main directory
    for folder_name in os.listdir(main_directory):
        patient_folder_path = os.path.join(main_directory, folder_name)
        
        # Process only directories
        if os.path.isdir(patient_folder_path):
            print(f"  - '{folder_name}' 폴더를 분석 중...")
            unique_dates = set()
            
            # Walk through all files in the patient's folder and its subfolders
            for root, _, files in os.walk(patient_folder_path):
                for filename in files:
                    # Find all date patterns in the filename
                    matches = date_pattern.findall(filename)
                    for date_str in matches:
                        unique_dates.add(date_str)
            
            follow_up_count = len(unique_dates)
            results.append({
                'Patient Folder': folder_name,
                'FollowUp_Count': follow_up_count
            })
            print(f"    -> 팔로업 횟수: {follow_up_count}회")

    if not results:
        print("분석할 환자 폴더를 찾지 못했습니다.")
        return

    # Create a pandas DataFrame and sort it
    df = pd.DataFrame(results)
    df = df.sort_values(by='FollowUp_Count', ascending=False)
    
    # Define the output Excel file path
    output_filename = f"FollowUp_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    output_path = os.path.join(main_directory, output_filename)
    
    try:
        # Save to Excel
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n분석 완료! 결과가 '{output_path}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"\n오류: 엑셀 파일 저장에 실패했습니다. 오류 메시지: {e}")


if __name__ == '__main__':
    # Get the main directory from user input
    top_level_folder = input("환자 폴더들이 들어있는 최상위 폴더의 경로를 입력하세요: ")
    
    if os.path.isdir(top_level_folder):
        count_followups_and_export_to_excel(top_level_folder)
    else:
        print(f"오류: 입력하신 경로 '{top_level_folder}'를 찾을 수 없거나 폴더가 아닙니다.")
