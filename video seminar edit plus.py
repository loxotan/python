import pandas as pd
import os
import subprocess

# ===== 기본 설정 =====
input_video = r"D:\Recording\20250926 이철수 879343 #16, 17 sinus & 1st.mp4"
output_dir = r"C:\Users\최수영\Desktop\참고자료\2025 동영상 세미나 가을"
excel_path = r"C:\Users\최수영\Desktop\참고자료\2025 동영상 세미나 가을\타임스탬프.xlsx"

# ===== 엑셀 읽기 =====
df = pd.read_excel(excel_path)

# note 열에서 배속이 있는 행만 필터링
speed_df = df[df['note'].astype(str).str.contains('x', case=False, na=False)]

# ===== 각 행에 대해 처리 =====
for idx, row in speed_df.iterrows():
    start = str(row['start']).strip()
    end = str(row['end']).strip()
    title = str(row['title']).strip()
    note = str(row['note']).lower()

    # 배속 인식
    if '2x' in note:
        speed_factor = 0.5
        speed_label = '(2x)'
    elif '5x' in note:
        speed_factor = 0.2
        speed_label = '(5x)'
    else:
        continue  # 기타 텍스트는 무시

    # 출력 파일명
    output_path = os.path.join(output_dir, f"{title} {speed_label}.mp4")

    # FFmpeg 필터 및 명령 생성
    cmd = [
        "ffmpeg",
        "-ss", start,
        "-to", end,
        "-i", input_video,
        "-filter:v", f"setpts={speed_factor}*PTS",
        "-an",
        "-c:v", "libx264",
        output_path
    ]

    # 명령 출력 및 실행
    print(" ".join(cmd))
    subprocess.run(cmd, shell=True)

print("\n=== 완료: note에 배속 표시가 있는 클립만 처리됨 ===")
