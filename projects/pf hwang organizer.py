import os
import shutil
from datetime import datetime
import calendar

# 요일을 한글로 변환하는 함수
def get_korean_weekday(weekday):
    korean_weekdays = ['월', '화', '수', '목', '금', '토', '일']
    return korean_weekdays[weekday]

def copy_and_rename_files(source_dir, target_root):
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            # 날짜 정보 파싱
            try:
                date = datetime.strptime(dir_name[:10], '%Y-%m-%d')
                week_day = get_korean_weekday(date.weekday())
                new_dir_name = date.strftime(f'%Y.%m.%d ({week_day})')
                new_target_dir = os.path.join(target_root, f'{date.year}년도', f'{date.month}월', new_dir_name)
                
                # 대상 폴더가 없으면 생성
                if not os.path.exists(new_target_dir):
                    os.makedirs(new_target_dir)
                
                # 파일 및 폴더 복사 (기존 폴더가 있다면, 예외를 처리하기 위한 추가 로직 필요)
                source_path = os.path.join(root, dir_name)
                target_path = os.path.join(new_target_dir, '최수영')
                if not os.path.exists(target_path):  # 대상 폴더가 존재하지 않는 경우에만 복사
                    shutil.copytree(source_path, target_path)
                    print(f'Copied and renamed {source_path} to {target_path}')
                else:
                    print(f'Target path {target_path} already exists. Consider handling this scenario.')
            except ValueError as e:
                print(f'Skipping {dir_name}: ', e)

# 사용 예시
source_dir = r'c:\Users\user\Desktop\날짜별'
target_root = r'c:\Users\user\Desktop\황날짜별'
copy_and_rename_files(source_dir, target_root)
