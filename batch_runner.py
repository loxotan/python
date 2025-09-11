# C:\SlicerScripts\batch_runner.py
import os, sys, subprocess, datetime

# ====== 사용자 설정 ======
BASE_DIR = r"C:\CT cut"                         # 환자 폴더들이 있는 상위 폴더
BAT_PATH = r"C:\SlicerScripts\run_slicer_pipeline.bat"  # 앞서 만든 배치 파일 경로
OUT_ROOT = r"C:\SlicerOut"                      # 환자별 출력 루트
USE_CPU  = False                                # CUDA 없는 PC면 True로
# ========================

def run_case(patient_dir):
    """환자 폴더(…\pre, …\post)를 찾아 배치 실행"""
    name = os.path.basename(patient_dir.rstrip(r"\/"))
    pre  = os.path.join(patient_dir, "pre")
    post = os.path.join(patient_dir, "post")
    if not os.path.isdir(pre) or not os.path.isdir(post):
        return f"[SKIP] {name}: 'pre' 또는 'post' 폴더가 없음"

    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    # 배치에서 읽을 환경변수 미리 주입 (BAT는 %PRE% %POST% %OUT%를 사용하도록 만들어 두었음)
    env = os.environ.copy()
    env["PRE"]  = pre
    env["POST"] = post
    env["OUT"]  = out_dir
    # CPU 강제 옵션 — 배치가 %EXTRA%를 쓰도록 해두었으면 반영됨
    env["EXTRA"] = "--cpu" if USE_CPU else ""

    print(f"[RUN] {name}")
    # cmd /c call 로 배치 호출 (경로에 공백·한글 있어도 안전)
    completed = subprocess.run(
        ['cmd', '/c', 'call', BAT_PATH],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # 실행 로그 저장
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"batch_runner_{stamp}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(completed.stdout)

    if completed.returncode == 0:
        return f"[OK]  {name} → {out_dir}"
    else:
        return f"[ERR] {name} (코드 {completed.returncode}) → 로그: {log_path}"

def main():
    if not os.path.isdir(BASE_DIR):
        print(f"[ERROR] BASE_DIR 없음: {BASE_DIR}")
        sys.exit(1)
    if not os.path.isfile(BAT_PATH):
        print(f"[ERROR] BAT_PATH 없음: {BAT_PATH}")
        sys.exit(1)
    os.makedirs(OUT_ROOT, exist_ok=True)

    patients = [d.path for d in os.scandir(BASE_DIR) if d.is_dir()]
    if not patients:
        print(f"[WARN] 환자 폴더가 비어 있음: {BASE_DIR}")
        return

    results = []
    for p in sorted(patients):
        results.append(run_case(p))

    print("\n=== Summary ===")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
