# 수술 영상 머리삭제기 v1.0 (PyTorch 버전)
# 이 스크립트는 수술 동영상에서 머리카락이 보이거나 초점이 맞지 않는 프레임을 제거합니다.

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# --- 설정 변수 ---
BASE_DATA_PATH = "c:\\python\\head remover data"
SAVE_FOLDER = os.path.join(BASE_DATA_PATH, "save")
REMOVE_FOLDER = os.path.join(BASE_DATA_PATH, "remove")
SPLIT_DATA_PATH = "c:\\python\\split_data"

# 이미지 크기 및 학습 파라미터
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 20 # 필요에 따라 조절
LEARNING_RATE = 0.001

# --- 1. 데이터 준비 (TensorFlow 버전과 동일) ---
def prepare_data():
    """
    원본 데이터 폴더에서 이미지를 읽어와 train, validation, test 폴더로 분배합니다.
    클래스 불균형을 고려하여 분할합니다.
    """
    print("--- 데이터 준비 시작 ---")
    
    if os.path.exists(SPLIT_DATA_PATH):
        shutil.rmtree(SPLIT_DATA_PATH)
        print(f"기존 '{SPLIT_DATA_PATH}' 폴더를 삭제했습니다.")

    save_files = [os.path.join(SAVE_FOLDER, f) for f in os.listdir(SAVE_FOLDER) if f.endswith(('jpg', 'png'))]
    remove_files = [os.path.join(REMOVE_FOLDER, f) for f in os.listdir(REMOVE_FOLDER) if f.endswith(('jpg', 'png'))]
    
    files = save_files + remove_files
    labels = [1] * len(save_files) + [0] * len(remove_files) # 1: save, 0: remove

    if not files:
        print("오류: 원본 데이터 폴더에 이미지가 없습니다.")
        return False, 0, 0

    X_train, X_temp, y_train, y_temp = train_test_split(files, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    datasets = {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
        'test': (X_test, y_test)
    }

    for split_name, (file_list, label_list) in datasets.items():
        for file_path, label in zip(file_list, label_list):
            class_name = 'save' if label == 1 else 'remove'
            target_dir = os.path.join(SPLIT_DATA_PATH, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(file_path, target_dir)
    
    print("--- 데이터 준비 완료 ---")
    print(f"훈련 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개, 테스트 데이터: {len(X_test)}개")
    return True, len(remove_files), len(save_files)

# --- 2. 모델 구축 및 훈련 (PyTorch) ---
def train_model_pytorch(num_remove, num_save):
    """
    PyTorch를 사용하여 전이학습 모델을 구축하고 훈련시킨 후 저장합니다.
    """
    print("\n--- PyTorch 모델 훈련 시작 ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 데이터 변환 및 증강
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
            transforms.CenterCrop((IMG_WIDTH, IMG_HEIGHT)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 데이터 로더
    image_datasets = {x: datasets.ImageFolder(os.path.join(SPLIT_DATA_PATH, x), data_transforms[x])
                      for x in ['train', 'validation']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if device.type == 'cuda' else 0)
                   for x in ['train', 'validation']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    
    # MobileNetV2 모델 로드
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # 기존 가중치 동결
    for param in model.parameters():
        param.requires_grad = False
        
    # 분류기 교체
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1) # 이진 분류를 위해 출력 1
    )
    model = model.to(device)

    # 손실 함수 (데이터 불균형 처리)
    # pos_weight: "remove" 클래스(0) 대비 "save" 클래스(1)의 비율
    pos_weight = torch.tensor([num_remove / num_save], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"BCEWithLogitsLoss pos_weight: {pos_weight.item():.4f}")

    # 옵티마이저 (새로운 분류기의 파라미터만 학습)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # 훈련 루프
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 모델 저장
    model_path = 'head_remover_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"--- 모델 훈련 완료 ---")
    print(f"훈련된 모델이 '{model_path}' 파일로 저장되었습니다.")

# --- 메인 실행 ---
if __name__ == '__main__':
    is_prepared, num_remove, num_save = prepare_data()
    if is_prepared:
        train_model_pytorch(num_remove, num_save)
