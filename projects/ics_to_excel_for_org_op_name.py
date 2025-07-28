import pandas as pd
from icalendar import Calendar
import re
from datetime import datetime
import os

# ics 파일 읽기 함수 정의
def parse_ics_to_excel(ics_file_path, excel_file_path):
    # 파일 경로에서 큰따옴표 제거 및 경로 정리
    ics_file_path = ics_file_path.strip('"')
    excel_file_path = excel_file_path.strip('"')

    # ics 파일 열기
    with open(ics_file_path, 'r', encoding='utf-8') as f:
        calendar = Calendar.from_ical(f.read())
    
    # 데이터 저장용 리스트 초기화
    data = []

    # 각 이벤트 정보 파싱
    for component in calendar.walk():
        if component.name == "VEVENT":
            dtstart = component.get('DTSTART').dt
            summary = component.get('SUMMARY')
            
            # 날짜만 추출
            date_str = dtstart.strftime('%Y-%m-%d') if isinstance(dtstart, datetime) else dtstart
            
            # summary 데이터를 공백 기준으로 나누기
            if summary:
                summary = re.sub(r',', ' ', summary)  # 쉼표를 공백으로 대체
                summary_parts = summary.split()
            else:
                summary_parts = []
            
            # 결과를 리스트에 추가
            data.append([date_str] + summary_parts)

    # 데이터프레임 생성 및 엑셀로 변환
    df = pd.DataFrame(data)
    df.columns = ['Date'] + [f'Summary_Part_{i+1}' for i in range(df.shape[1] - 1)]
    
    # 날짜 기준으로 정렬
    df['Date'] = pd.to_datetime(df['Date'])  # 문자열을 datetime으로 변환
    today = datetime.today().date()
    df = df[df['Date'].dt.date <= today]     # 오늘 이후 날짜 제외
    df = df.sort_values(by='Date')          # 날짜 오름차순 정렬
    
    df.to_excel(excel_file_path, index=False, engine='openpyxl')
    
    print(f"엑셀 파일이 '{excel_file_path}'에 저장되었습니다.")

# 파일 경로 설정 및 함수 호출
if __name__ == "__main__":
    ics_file = input("ICS 파일의 경로를 입력하세요: ")
    excel_file = input("저장할 엑셀 파일의 경로를 입력하세요: ")
    parse_ics_to_excel(ics_file, excel_file)

#"C:\Users\user\Downloads\lodicaine@gmail.com.ical (3)\7f492564106b9c612167726dcce90d3a2d56178e89d3d76c7a798f7843ab6a5b@group.calendar.google.com.ics"
#
