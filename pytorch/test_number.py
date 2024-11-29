import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import glob
import shutil

# 디바이스 설정 (CPU 사용)
device = torch.device("cpu")

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기를 64x64로 변환
    transforms.ToTensor(),         # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 테스트 데이터셋 로드
test_data_dir = r'c:/Users/user/Desktop/test_dataset'  # 테스트 데이터가 저장된 폴더 경로
test_image_paths = glob.glob(os.path.join(test_data_dir, '*.*'))  # 폴더 내 모든 이미지 파일 경로 가져오기

# 모델 정의 (훈련했던 것과 동일한 구조)
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
        x = x.view(-1, 64 * 14 * 14)  # Flatten the tensor for the fully connected layer
        x = torch.relu(self.fc1(x))  # Correctly pass x to fc1 and then apply ReLU
        x = torch.sigmoid(self.fc2(x))  # Correctly pass x to fc2 and then apply Sigmoid
        return x

# 모델 로드
model = SimpleBinaryCNN().to(device)
model.load_state_dict(torch.load('c:/Users/user/Desktop/number_detection_model.pth', map_location=device))

model.eval()

# 테스트 데이터 예측 및 분류
detection_results = {"number_present": [], "number_absent": []}

with torch.no_grad():
    for image_path in test_image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        outputs = model(image)
        predicted = (outputs > 0.5).float()

        # 예측 결과 반전 (number_present와 number_absent가 바뀐 경우)
        if predicted.item() == 0.0:
            detection_results["number_present"].append(image_path)
        else:
            detection_results["number_absent"].append(image_path)

# 결과 출력 및 이미지 이동
number_present_dir = os.path.join(test_data_dir, 'number_present')
number_absent_dir = os.path.join(test_data_dir, 'number_absent')

# 폴더가 없으면 생성
os.makedirs(number_present_dir, exist_ok=True)
os.makedirs(number_absent_dir, exist_ok=True)

# 이미지 이동
for path in detection_results["number_present"]:
    shutil.move(path, os.path.join(number_present_dir, os.path.basename(path)))

for path in detection_results["number_absent"]:
    shutil.move(path, os.path.join(number_absent_dir, os.path.basename(path)))

# 최종 결과 출력
print("Images with numbers:")
for path in detection_results["number_present"]:
    print(os.path.join(number_present_dir, os.path.basename(path)))

print("\nImages without numbers:")
for path in detection_results["number_absent"]:
    print(os.path.join(number_absent_dir, os.path.basename(path)))
