import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 디바이스 설정 (CPU 사용)
device = torch.device("cpu")

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((64, 64)),      # 이미지 크기를 64x64로 변환
    transforms.ToTensor(),            # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 데이터셋 로드
data_dir = r'c:/Users/user/Desktop/dataset'  # 데이터가 저장된 폴더 경로
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# 간단한 CNN 모델 정의
class SimpleBinaryCNN(nn.Module):
    def __init__(self):
        super(SimpleBinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # 3채널(RGB) 입력, 32개의 3x3 필터
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # 이미지 크기(64x64 기준) 조정에 맞게 설정
        self.fc2 = nn.Linear(128, 1)  # 이진 분류이므로 출력 노드는 1개

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # 2x2 맥스풀링
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 14 * 14)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 이진 분류에서 sigmoid 사용
        return x

# 모델, 손실 함수 및 최적화기 설정
model = SimpleBinaryCNN().to(device)
criterion = nn.BCELoss()  # 이진 분류 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 조기 종료 설정
patience = 3  # 손실 값이 개선되지 않는 횟수
min_loss = float('inf')  # 손실 값 초기화
trigger_times = 0  # 개선되지 않는 에포크 수

# 모델 학습
if __name__ == "__main__":
    for epoch in range(10):  # 최대 10 epoch 동안 학습
        model.train()
        epoch_loss = 0  # 에포크별 손실 값
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # 데이터 CPU로 이동
            optimizer.zero_grad()

            # 순전파 및 손실 계산
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/10], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 에포크 평균 손실 계산
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/10] - Average Loss: {epoch_loss:.4f}")

        # 조기 종료 조건 확인
        if epoch_loss < 0.0001:
            print("Loss below 0.0001, stopping training early.")
            break

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            trigger_times = 0  # 손실이 감소하면 초기화
        else:
            trigger_times += 1  # 손실이 개선되지 않으면 증가
            print(f"No improvement in loss for {trigger_times} epochs")

        if trigger_times >= patience:  # 개선되지 않는 횟수가 한계에 도달
            print("Early stopping triggered")
            break

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), r'c:/Users/user/Desktop/number_detection_model.pth')

# 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # 데이터 CPU로 이동
        outputs = model(images)
        predicted = (outputs > 0.5).float()  # 0.5 기준으로 클래스 예측
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
