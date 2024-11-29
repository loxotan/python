import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pandas as pd
from PIL import Image
import argparse

# Dataset 클래스 정의
class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    self.image_paths.append(image_path)
                    name, number = label.split(' ')
                    self.labels.append((int(number), name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        labels_number, labels_name = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # labels_number를 텐서로 변환
        labels_number = torch.tensor(labels_number, dtype=torch.long)
        
        return image, (labels_number, labels_name)

# 모델 정의 (ResNet18 기반)
class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)  # 특징을 줄여준다
        
        # 각 자리수별 분류기 (최대 7자리 숫자)
        self.fc_digits = nn.ModuleList([nn.Linear(128, 10) for _ in range(7)])  # 각 자리수가 0-9 사이의 값을 가짐
        self.fc_name = nn.Linear(128, 2350)  # 한글 2350자 분류 (2글자~7글자)
    
    def forward(self, x):
        x = self.backbone(x)
        
        # 각 자리수별로 예측값 생성
        digit_outputs = [fc_digit(x) for fc_digit in self.fc_digits]
        name = self.fc_name(x)
        
        return digit_outputs, name

# 엑셀 파일을 읽어 딕셔너리로 저장하는 함수
def excel_list_up(excel_path):
    df = pd.read_excel(excel_path)
    id_name_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return id_name_dict

# 문자열을 인덱스로 변환하는 함수 (예시로 간단한 매핑 제공)
def name_to_index(name, max_length=7):
    indices = [min(ord(ch) - ord('가'), 2349) for ch in name]  # 한글 유니코드 기준으로 인덱스 변환, 범위를 0-2349로 제한
    indices += [0] * (max_length - len(indices))  # 패딩 처리
    return indices

# 모델 학습 함수
def train_model(data_dir, excel_path):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = OCRDataset(train_dir, transform=transform)
    test_dataset = OCRDataset(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRModel().to(device)
    criterion_digit = nn.CrossEntropyLoss().to(device)
    criterion_name = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()  # Mixed precision training을 위한 GradScaler
    else:
        scaler = None

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels_number, labels_name = labels
            
            # labels_number를 각 자리수로 분리하여 텐서로 만듭니다.
            try:
                labels_digits = [[int(d) for d in str(number).zfill(7)] for number in labels_number]
            except ValueError as e:
                print(f"Skipping invalid label: {labels_number}")
                continue
            labels_digits = torch.tensor(labels_digits, dtype=torch.long).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast() if scaler else torch.no_grad():  # Mixed precision을 사용하여 forward pass
                digit_outputs, name_out = model(images)
                
                # 각 자리수별 손실 계산
                loss_digits = 0
                for i in range(7):
                    loss_digits += criterion_digit(digit_outputs[i], labels_digits[:, i])
                
                # 이름에 대한 손실 계산
                labels_name_tensor = torch.tensor([name_to_index(name) for name in labels_name], dtype=torch.long).to(device)
                loss_name = criterion_name(name_out, labels_name_tensor[:, 0])

                loss = loss_digits + loss_name

            if scaler:
                scaler.scale(loss).backward()  # Mixed precision을 사용하여 backward pass
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # 테스트 데이터셋에서 정확도 측정
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels_number, labels_name = labels
                
                # 테스트 데이터에 대해서도 동일한 처리
                labels_digits = [[int(d) for d in str(number).zfill(7)] for number in labels_number]
                labels_digits = torch.tensor(labels_digits, dtype=torch.long).to(device)
                labels_name_tensor = torch.tensor([name_to_index(name) for name in labels_name], dtype=torch.long).to(device)
                
                with torch.cuda.amp.autocast() if scaler else torch.no_grad():  # Mixed precision을 사용하여 inference
                    digit_outputs, name_out = model(images)
                
                # 각 자리수별 예측값과 정답 비교
                for i in range(7):
                    _, predicted_digit = torch.max(digit_outputs[i], 1)
                    correct += (predicted_digit == labels_digits[:, i]).sum().item()
                
                # 이름 예측값과 정답 비교
                predicted_name = torch.argmax(name_out, dim=1)  # 각 문자의 예측값을 얻음
                correct += (predicted_name == labels_name_tensor[:, 0]).sum().item()

                total += images.size(0) * 8  # 총 7자리 숫자 + 이름 비교 (8개의 예측값)

        accuracy = 100 * correct / total
        print(f"Test Accuracy after Epoch {epoch+1}: {accuracy}%")

    # 학습된 모델 저장
    torch.save(model.state_dict(), "/home/user/recognition_model_2.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data directory")
    parser.add_argument('--excel_path', type=str, required=True, help="Path to the excel file")
    args = parser.parse_args()

    train_model(args.data_dir, args.excel_path)
