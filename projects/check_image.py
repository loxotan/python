from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def crop_non_white_areas(image_path):
    # 이미지 열기
    image = Image.open(image_path)
    
    # 이미지를 RGB로 변환
    image_rgb = image.convert('RGB')
    
    # 이미지를 numpy 배열로 변환
    image_array = np.array(image_rgb)
    
    # 흰색 배경을 제외한 마스크 생성
    mask = (image_array[:,:,0] < 250) | (image_array[:,:,1] < 250) | (image_array[:,:,2] < 250)
    
    # 마스크가 적용된 영역의 좌표 가져오기
    coords = np.argwhere(mask)
    
    # 마스크가 비어 있으면 예외 처리
    if coords.size == 0:
        print("이미지에서 비어있지 않은 영역을 찾을 수 없습니다.")
        return None
    
    # 마스크의 가장자리 좌표 찾기
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 크롭된 영역의 이미지 생성
    cropped_array = image_array[y_min:y_max+1, x_min:x_max+1]
    cropped_image = Image.fromarray(cropped_array)
    
    # 결과 이미지 저장
    cropped_image_path = image_path.replace('.png', '_cropped.png')
    cropped_image.save(cropped_image_path)
    print(f"크롭된 이미지가 저장되었습니다: {cropped_image_path}")

    # 결과 이미지 표시
    cropped_image.show()

# 사용 예시
crop_non_white_areas(r"C:\Users\user\Desktop\날짜별\2024\2024-07-05 [17]-\DSC_5392 2024-07-05.JPG")
# 크롭된 이미지 보여주기
