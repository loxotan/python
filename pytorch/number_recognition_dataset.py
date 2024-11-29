import os
import shutil
import random

def random_destination(dataset):
    """
    랜덤하게 destination 폴더를 반환하는 함수
    0.2 미만이면 test, 그렇지 않으면 train을 반환합니다.
    """
    if random.random() < 0.2:
        return os.path.join(dataset, 'test')
    else:
        return os.path.join(dataset, 'train')

def extract_first_file(indiv_source, indiv_destination):
    """
    소스 폴더에서 첫 번째 파일을 목적지 폴더로 복사하는 함수
    """
    if not os.path.exists(indiv_source):
        print(f"Source folder does not exist: {indiv_source}")
        return

    files = sorted(os.listdir(indiv_source))
    if files:
        first_file = files[0]
        source_file_path = os.path.join(indiv_source, first_file)
        os.makedirs(indiv_destination, exist_ok=True)
        shutil.copy(source_file_path, indiv_destination)
        print(f"Copied {first_file} to {indiv_destination}")
    else:
        print(f"No files found in source folder: {indiv_source}")

def make_dataset(dataset_destination):
    """
    dataset 폴더 안에 '환자명 고유번호' 폴더가 있는지 확인하고,
    없으면 생성하는 함수
    """
    if not os.path.exists(dataset_destination):
        os.makedirs(dataset_destination)
        print(f"Created folder: {dataset_destination}")
    else:
        print(f"Folder already exists: {dataset_destination}")

def main():
    """
    메인 함수: 각 날짜별 소스 폴더에서 첫 번째 파일을 test 또는 train으로 분배
    """
    base_source = r'E:/IO photo/날짜별'
    dataset = r'C:/Users/user/Desktop/숫자 확인 모델'
    
    if not os.path.exists(base_source):
        print(f"Base source folder does not exist: {base_source}")
        return

    # 년도별 폴더 순회
    for year_folder in os.listdir(base_source):
        year_path = os.path.join(base_source, year_folder)
        if os.path.isdir(year_path):
            # 날짜별 폴더 순회
            for date_folder in os.listdir(year_path):
                date_path = os.path.join(year_path, date_folder)
                if os.path.isdir(date_path):
                    # 각 '환자명 고유번호 날짜' 폴더를 처리
                    for patient_folder in os.listdir(date_path):
                        indiv_source = os.path.join(date_path, patient_folder)
                        indiv_destination = os.path.join(random_destination(dataset), ' '.join(patient_folder.split()[:2]))
                        make_dataset(indiv_destination)
                        extract_first_file(indiv_source, indiv_destination)

if __name__ == "__main__":
    main()
