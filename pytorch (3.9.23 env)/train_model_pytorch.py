import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time

# --- 1. 설정값 (Configuration) ---
# 데이터셋이 있는 상위 폴더 경로를 지정하세요.
DATA_DIR = 'dataset'

# 모델 학습을 위한 파라미터
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 # 초기 학습 횟수, 필요시 조절

# GPU 사용 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- 2. 데이터셋 로드 및 준비 ---
def load_datasets():
    """데이터셋 폴더에서 훈련, 검증, 테스트 데이터를 로드합니다."""
    
    # 이미지 변환: 리사이즈, 텐서 변환, 정규화
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dir = os.path.join(DATA_DIR, 'train')
    validation_dir = os.path.join(DATA_DIR, 'validation')
    test_dir = os.path.join(DATA_DIR, 'test')

    if not all(os.path.exists(d) for d in [train_dir, validation_dir, test_dir]):
        print(f"Error: Make sure the dataset directory structure is correct in '{DATA_DIR}'")
        print("Expected structure: dataset/{train, validation, test}/{patient_info, clinical}")
        return None, None

    # ImageFolder를 사용하여 데이터셋 생성
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(validation_dir, data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, data_transforms['val']) # Test는 val 변환 사용
    }

    # DataLoader 생성
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    return dataloaders, class_names, dataset_sizes

# --- 3. 모델 구축 (전이 학습) ---
def build_model(num_classes):
    """MobileNetV2를 기반으로 한 전이 학습 모델을 구축합니다."""
    # 사전 학습된 MobileNetV2 모델 로드
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # 기존 가중치 동결
    for param in model.parameters():
        param.requires_grad = False

    # 새로운 분류층으로 교체
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 1) # 이진 분류이므로 출력은 1
    )
    
    return model.to(DEVICE)

# --- 4. 모델 학습 함수 ---
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    """모델을 학습하고 검증하는 메인 루프입니다."""
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 각 에포크는 훈련과 검증 단계를 가짐
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 훈련 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().view(-1, 1)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    loss = criterion(outputs, labels)

                    # 훈련 단계에서만 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return model, history

# --- 5. 시각화 함수 ---
def plot_history(history):
    """모델 학습 과정의 정확도와 손실을 그래프로 출력합니다."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# --- 6. 메인 실행 로직 ---
def main():
    # 데이터 로드
    dataloaders, class_names, dataset_sizes = load_datasets()
    if dataloaders is None:
        return

    # 모델 구축
    model = build_model(len(class_names))

    # 손실 함수와 옵티마이저 정의
    # BCEWithLogitsLoss는 Sigmoid와 BCELoss를 합친 것으로, 수치적으로 안정적임
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 모델 학습
    model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=EPOCHS)

    # 결과 시각화
    plot_history(history)

    # 최종 모델 평가 (테스트 데이터)
    print("\n--- Evaluating Model on Test Set ---")
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().view(-1, 1)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            test_corrects += torch.sum(preds == labels.data)
    
    test_acc = test_corrects.double() / dataset_sizes['test']
    print(f'Test Accuracy: {test_acc:.4f}')

    # 학습된 모델의 state_dict 저장
    torch.save(model.state_dict(), 'photo_classifier_model.pth')
    print("\nModel state_dict saved to photo_classifier_model.pth")

if __name__ == '__main__':
    main()
