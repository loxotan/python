import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 기본 경로 설정
BASE_SOURCE = r"C:\Users\user\Desktop\날짜별"
NUMBER_DETECTION_MODEL_PATH = r"C:\Users\user\Desktop\number_detection_model.pth"
RECOGNITION_MODEL_PATH = r"C:\Users\user\Desktop\숫자 확인 모델\recognition_model.pth"
EXCEL_PATH = r"C:\Users\user\Desktop\숫자 확인 모델\patient_list.xlsx"

device = torch.device("cpu")

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기를 64x64로 변환
    transforms.ToTensor(),         # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 숫자 인식 모델 클래스
class SimpleBinaryCNN(nn.Module):
    def __init__(self):
        super(SimpleBinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 모델 로드
number_detection_model = SimpleBinaryCNN().to(device)
number_detection_model.load_state_dict(torch.load(NUMBER_DETECTION_MODEL_PATH, map_location=device))
number_detection_model.eval()

def number_detection(folder_path):
    detection_results = []
    with torch.no_grad():
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            outputs = number_detection_model(image)
            predicted = (outputs > 0.5).float()
            if predicted.item() == 0.0:
                detection_results.append(image_name)
    return detection_results

# 환자 리스트 불러오기
def load_patient_list():
    df = pd.read_excel(EXCEL_PATH)
    patient_dict = {str(row.iloc[0]): row.iloc[1] for _, row in df.iterrows()}
    return patient_dict

# 이미지 예측 및 라벨링
def predict_image(image_path, label_encoder, patient_dict):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_classes = len(label_encoder.classes_)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 모델의 state_dict 불러오기 (fc 레이어 제외)
    state_dict = torch.load(RECOGNITION_MODEL_PATH, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()

    transform_recognition = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform_recognition(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        unique_number = label_encoder.inverse_transform([predicted.item()])[0]
        patient_name = patient_dict[unique_number]
    return f"{patient_name} {unique_number}"

# 사용자에게 이미지 보여주기
def show_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# 메인 함수
def main():
    # 파트 1: 숫자 인식 모델로 환자 정보 포함 이미지 분류
    number_detected = []
    for year_folder in os.listdir(BASE_SOURCE):
        year_path = os.path.join(BASE_SOURCE, year_folder)
        if os.path.isdir(year_path):
            for date_folder in os.listdir(year_path):
                if '-=' not in date_folder:
                    date_path = os.path.join(year_path, date_folder)
                    detected_images = number_detection(date_path)
                    number_detected.extend([(date_path, img) for img in detected_images])

    # 파트 2: 고유번호와 이름 확인 후 폴더 정리
    patient_dict = load_patient_list()
    label_encoder = LabelEncoder()
    label_encoder.fit(list(patient_dict.keys()))

    for (date_path, image_name) in number_detected:
        image_path = os.path.join(date_path, image_name)
        predicted_label = predict_image(image_path, label_encoder, patient_dict)

        # 사용자에게 예측 결과 확인
        while True:
            user_input = input(f"{predicted_label} 이 맞습니까? 맞으면 엔터를 누르시고, 아니면 아무 숫자나 입력하세요: ")
            # 이미지 보여주기
            show_image(image_path)
            if user_input == "":
                break
            else:
                predicted_label = input("이름과 고유번호를 입력하세요 (예: 김철수 1234123): ")
                break

        # 폴더 생성 및 파일 이동
        new_folder_name = f"{predicted_label} {date_folder}"
        new_folder_path = os.path.join(date_path, new_folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        for file_name in os.listdir(date_path):
            if file_name == image_name:
                continue
            file_path = os.path.join(date_path, file_name)
            if file_name in [img for _, img in number_detected]:
                break
            shutil.move(file_path, new_folder_path)

if __name__ == "__main__":
    main()
