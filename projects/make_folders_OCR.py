import os
import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# Tesseract 설치 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

source_dir = r'C:\Users\user\Desktop\날짜별'
output_dir = r'C:\Users\user\Desktop\output_dataset'

name_pattern = re.compile(r'[가-힣]{2,8}')
patient_number_pattern = re.compile(r'\d{4,7}')

def extract_patient_info(image_path):
    try:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)
        
        # 이미지 전처리: 흑백 변환, 대비 조정, 샤프닝 등
        image = image.convert('L')
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageEnhance.Contrast(image).enhance(2)
        image = ImageEnhance.Sharpness(image).enhance(2)
        image = image.filter(ImageFilter.GaussianBlur(1))
        
        # 전처리된 이미지를 저장
        converted_path = image_path.replace('.JPG', '_converted.jpg')
        image.save(converted_path, format='JPEG')
        print(f"Image converted and saved to: {converted_path}")
        
        # OCR 옵션 설정
        custom_config = r'--oem 1 --psm 11'  # --psm 11은 이미지의 텍스트 블록을 찾는 데 도움
        text = pytesseract.image_to_string(image, lang='kor', config=custom_config)
        print(f"Extracted text: {text}")
        
        # 정규 표현식을 사용하여 이름과 환자 번호 추출
        name = name_pattern.search(text)
        patient_number = patient_number_pattern.search(text)
        
        if name and patient_number:
            os.remove(converted_path)
            return name.group(2), patient_number.group(1)
        else:
            os.remove(converted_path)
            return None
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

for year in os.listdir(source_dir):
    year_path = os.path.join(source_dir, year)
    if os.path.isdir(year_path):
        print(f"Entering year directory: {year_path}")
        for date_folder in os.listdir(year_path):
            date_path = os.path.join(year_path, date_folder)
            if os.path.isdir(date_path):
                print(f"Entering date directory: {date_path}")
                files_in_dir = os.listdir(date_path)
                print(f"Files in {date_path}: {files_in_dir}")

                for image_file in sorted(files_in_dir):
                    if "_converted" in image_file:
                        continue
                    
                    image_path = os.path.join(date_path, image_file)
                    
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.mpo')):
                        print(f"Found image file: {image_file}")
                        patient_info = extract_patient_info(image_path)
                        if patient_info:
                            patient_name, patient_number = patient_info
                            patient_dir = os.path.join(output_dir, f"{patient_name} {patient_number} {date_folder}")
                            try:
                                os.makedirs(patient_dir, exist_ok=True)
                                print(f"Created directory: {patient_dir}")
                            except Exception as e:
                                print(f"Error creating directory {patient_dir}: {str(e)}")

print("Processing completed.")
