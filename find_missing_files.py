import os
import sys

def get_relative_paths(directory):
    """Walks a directory and returns a set of relative file paths."""
    file_paths = set()
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(full_path, directory)
            file_paths.add(relative_path)
    return file_paths

if __name__ == "__main__":
    # --- 경로를 여기에 직접 입력하세요 ---
    # 앞에 r을 붙이면 경로에 있는 \를 문자로 인식해서 편리합니다.
    source_dir = r"\\RaiDrive-최수영\Synology\usbshare1\photo\USB Copy_2025-08-17_000001\Photo (사진만.. 날짜 순대로 해주세요)"
    dest_dir = r"\\RaiDrive-최수영\Synology\perio-photo\Photo (사진만.. 날짜 순대로 해주세요)"
    # ------------------------------------

    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")

    if not os.path.isdir(source_dir):
        print(f"Error: Source path does not exist or is not a directory.")
        sys.exit(1)
    if not os.path.isdir(dest_dir):
        print(f"Error: Destination path does not exist or is not a directory.")
        sys.exit(1)

    print(f"\nScanning source directory... (This may take a while)")
    source_files = get_relative_paths(source_dir)
    print(f"Found {len(source_files)} files in source.")

    print(f"\nScanning destination directory... (This may take a while)")
    dest_files = get_relative_paths(dest_dir)
    print(f"Found {len(dest_files)} files in destination.")

    missing_files = source_files - dest_files

    if not missing_files:
        print("\nSuccess: No missing files found. The destination is a perfect copy.")
    else:
        print(f"\nFound {len(missing_files)} missing files that are in source but not in destination:")
        # Sort for cleaner output
        for missing_file in sorted(list(missing_files)):
            print(missing_file)