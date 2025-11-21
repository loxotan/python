import pandas as pd
import os
import subprocess

# ===== 설정 =====
input_video = r"D:\Recording\20250926 이철수 879343 #16, 17 sinus & 1st.mp4"
output_dir = r"C:\Users\최수영\Desktop\참고자료\2025 동영상 세미나 가을"
excel_path = r"C:\Users\최수영\Desktop\참고자료\2025 동영상 세미나 가을\타임스탬프.xlsx"

# ===== 엑셀 불러오기 =====
df = pd.read_excel(excel_path)

# ===== 각 행 처리 =====
for idx, row in df.iterrows():
    start = row['start']
    end = row['end']
    title = str(row['title']).strip()
    note = str(row.get('note', '')).strip().lower()

    # 출력 파일명
    output_path = os.path.join(output_dir, f"{title}.mp4")

    # 배속 처리
    if '2x' in note:
        vf_filter = "-vf setpts=0.5*PTS"
    elif '5x' in note:
        vf_filter = "-vf setpts=0.2*PTS"
    else:
        vf_filter = ""

    # FFmpeg 명령
    cmd = f'ffmpeg -i "{input_video}" -ss {start} -to {end} {vf_filter} -c:v libx264 -c:a aac -strict -2 "{output_path}"'
    print(cmd)

    # 실행 (주석 해제 시 실제 실행됨)
    subprocess.run(cmd, shell=True)
