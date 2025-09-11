import os
import re

def batch_rename_photos():
    """
    Prompts the user for a folder path, including all subfolders, and renames files
    from "Name YYYY-MM-DD.ext" to "YYYY-MM-DD Name.ext".
    Handles quoted paths.
    """
    folder_path = input("사진 파일이 있는 폴더의 전체 경로를 입력하세요: ").strip().strip('"')

    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}'는 유효한 폴더가 아닙니다.")
        return

    # 파일 이름을 파싱하기 위한 정규표현식: (이름 부분) (날짜 부분)(확장자 부분)
    # 예: "DSC_1234 2025-08-06.jpg"
    pattern = re.compile(r"(.+?)\s+(\d{4}-\d{2}-\d{2})(\.\w+)$")

    renamed_count = 0
    skipped_count = 0

    print(f"\n'{folder_path}' 및 모든 하위 폴더를 스캔합니다...")

    for root, _, files in os.walk(folder_path):
        for filename in files:
            old_full_path = os.path.join(root, filename)

            match = pattern.match(filename)

            if match:
                # 정규표현식에 매칭되면 이름, 날짜, 확장자 추출
                name_part = match.group(1).strip()
                date_part = match.group(2)
                extension_part = match.group(3)

                # 새로운 파일 이름 조합: "YYYY-MM-DD Name.ext"
                new_filename = f"{date_part} {name_part}{extension_part}"
                new_full_path = os.path.join(root, new_filename)

                # 만약 새로운 파일 이름이 이미 존재하면 건너뛰기
                if os.path.exists(new_full_path):
                    print(f"SKIPPED: '{new_filename}' 파일이 이미 존재합니다. (in {root})")
                    skipped_count += 1
                    continue

                # 파일 이름 변경 실행
                try:
                    os.rename(old_full_path, new_full_path)
                    print(f"RENAMED: '{filename}' -> '{new_filename}' (in {root})")
                    renamed_count += 1
                except OSError as e:
                    print(f"ERROR: '{filename}' 파일 이름 변경 중 오류 발생: {e}")
                    skipped_count += 1
            else:
                # 패턴에 맞지 않는 파일은 건너뛰기
                skipped_count += 1
    
    print("\n--- 작업 완료 ---")
    print(f"총 {renamed_count}개의 파일 이름을 변경했습니다.")
    print(f"{skipped_count}개의 파일을 건너뛰었습니다.")

if __name__ == "__main__":
    batch_rename_photos()