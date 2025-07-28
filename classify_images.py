import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import shutil # 파일 이동을 위해 추가

# --- 1. 설정값 (Configuration) ---
# 분류하고 싶은 이미지가 있는 폴더 경로를 지정하세요.
UNCLASSIFIED_IMAGE_DIR = r"C:\Users\최수영\Desktop\data\106D7500" 

# 분류된 이미지를 저장할 상위 폴더 경로를 지정하세요.
CLASSIFIED_OUTPUT_DIR = r"C:\Users\최수영\Desktop\data"

IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 모델 아키텍처 재구성 ---
def build_model(num_classes):
    """MobileNetV2를 기반으로 한 전이 학습 모델을 구축합니다."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # 기존 가중치 동결
    for param in model.parameters():
        param.requires_grad = False

    # 새로운 분류층으로 교체 (train_model_pytorch.py와 동일하게)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 1) # 이진 분류이므로 출력은 1
    )
    
    return model.to(DEVICE)

# --- 3. 이미지 전처리 정의 (학습 시 val transforms와 동일) ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. 메인 분류 로직 ---
def main():
    print(f"Using device: {DEVICE}")

    # 모델 로드
    # num_classes는 학습 시 사용한 클래스 수 (여기서는 info, not_info 2개)
    model = build_model(num_classes=2) 
    
    # 학습된 가중치 불러오기
    model_path = 'photo_classifier_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # 모델을 평가 모드로 설정

    print(f"\nClassifying images in: {UNCLASSIFIED_IMAGE_DIR}")
    if not os.path.exists(UNCLASSIFIED_IMAGE_DIR):
        print(f"Error: Unclassified image directory '{UNCLASSIFIED_IMAGE_DIR}' not found.")
        print("Please create this directory and put your images inside, or change the path.")
        return

    # 분류할 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(UNCLASSIFIED_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print("No image files found in the specified directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(UNCLASSIFIED_IMAGE_DIR, image_file)
        
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # 배치 차원 추가

            with torch.no_grad(): # 추론 시에는 기울기 계산 비활성화
                input_batch = input_batch.to(DEVICE)
                output = model(input_batch)
                
                # Sigmoid를 통과시켜 확률로 변환하고 0.5 기준으로 분류
                prediction = torch.sigmoid(output) > 0.5
                
                predicted_class = "not_info" if prediction.item() else "info"
                print(f"Image: {image_file} -> Predicted Class: {predicted_class}")

                # 예측된 클래스 폴더 생성 및 이미지 이동
                output_class_dir = os.path.join(CLASSIFIED_OUTPUT_DIR, predicted_class)
                os.makedirs(output_class_dir, exist_ok=True) # 폴더가 없으면 생성
                
                destination_path = os.path.join(output_class_dir, image_file)
                shutil.move(image_path, destination_path)
                print(f"Moved to: {destination_path}")

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

if __name__ == '__main__':
    main()
