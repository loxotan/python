import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import shutil # 파일 복사를 위해 추가
from sklearn.model_selection import train_test_split # 데이터 분할을 위해 추가
import torch.optim.lr_scheduler # 학습률 스케줄러를 위해 추가

# --- 1. 설정값 (Configuration) ---

# 학습률 스케줄러 설정
LR_SCHEDULER_PATIENCE = 5 # 검증 손실이 이 에포크 동안 개선되지 않으면 학습률 감소
LR_SCHEDULER_FACTOR = 0.1 # 학습률 감소 비율

# 조기 종료 설정
EARLY_STOPPING_PATIENCE = 10 # 검증 손실이 이 에포크 동안 개선되지 않으면 학습 중단
MIN_DELTA = 0.0001 # 조기 종료를 위한 최소 개선치

# 데이터셋이 있는 상위 폴더 경로를 지정하세요.
DATA_DIR = 'dataset'

# 원본 데이터셋 폴더 경로 (사용자가 실제 info와 not_info 폴더가 있는 상위 경로로 변경해야 합니다!)
ORIGINAL_DATA_ROOT = r"C:\Users\최수영\Desktop\data" 

# 분할 비율
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 # 나머지

# 모델 학습을 위한 파라미터
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50 # 초기 학습 횟수, 필요시 조절

# GPU 사용 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. 데이터셋 로드 및 준비 ---
def load_datasets():
    """데이터셋 폴더에서 훈련, 검증, 테스트 데이터를 로드합니다."""

    # 데이터 분할 및 복사 로직 추가
    # 이미 분할된 데이터가 존재하면 다시 분할하지 않음
    # (dataset/train/info 폴더가 존재하면 이미 분할된 것으로 간주)
    if not os.path.exists(os.path.join(DATA_DIR, 'train', 'info')):
        print("데이터셋 분할 및 복사 중...")
        
        # 기존 dataset 폴더 삭제 (선택 사항, 재실행 시 깨끗하게 시작)
        if os.path.exists(DATA_DIR):
            print(f"기존 '{DATA_DIR}' 폴더를 삭제합니다.")
            shutil.rmtree(DATA_DIR)
        
        # 필요한 디렉토리 생성
        for phase in ['train', 'validation', 'test']:
            os.makedirs(os.path.join(DATA_DIR, phase, 'info'), exist_ok=True)
            os.makedirs(os.path.join(DATA_DIR, phase, 'not_info'), exist_ok=True)

        for class_name in ['info', 'not_info']:
            original_class_dir = os.path.join(ORIGINAL_DATA_ROOT, class_name)
            if not os.path.exists(original_class_dir):
                print(f"Error: Original data directory '{original_class_dir}' not found.")
                print(f"Please set ORIGINAL_DATA_ROOT to the correct path where 'info' and 'not_info' folders are located.")
                return None, None, None

            images = [os.path.join(original_class_dir, f) for f in os.listdir(original_class_dir) if os.path.isfile(os.path.join(original_class_dir, f))]
            
            if not images:
                print(f"Warning: No images found in '{original_class_dir}'. Skipping this class.")
                continue

            # train, temp (val + test) 분할
            train_images, temp_images = train_test_split(images, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
            
            # val, test 분할
            # test_size는 temp_images에서 test_ratio만큼을 의미하므로 비율을 조정
            val_images, test_images = train_test_split(temp_images, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), random_state=42)

            print(f"Class: {class_name}")
            print(f"  Train: {len(train_images)} images")
            print(f"  Validation: {len(val_images)} images")
            print(f"  Test: {len(test_images)} images")

            # 파일 복사
            for img_path in train_images:
                shutil.copy(img_path, os.path.join(DATA_DIR, 'train', class_name))
            for img_path in val_images:
                shutil.copy(img_path, os.path.join(DATA_DIR, 'validation', class_name))
            for img_path in test_images:
                shutil.copy(img_path, os.path.join(DATA_DIR, 'test', class_name))
        print("데이터셋 분할 및 복사 완료.")
    else:
        print("데이터셋이 이미 분할되어 있습니다. 기존 데이터를 사용합니다.")
    
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

    # 데이터 분할 로직이 성공적으로 실행되었는지 다시 확인
    if not all(os.path.exists(d) for d in [train_dir, validation_dir, test_dir]):
        print(f"Error: Dataset directories were not created correctly in '{DATA_DIR}'")
        return None, None, None

    # ImageFolder를 사용하여 데이터셋 생성
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(validation_dir, data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, data_transforms['val']) # Test는 val 변환 사용
    }

    # DataLoader 생성
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True) # pin_memory=True 추가
        for x in ['train', 'val']
    }
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True) # pin_memory=True 추가

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    return dataloaders, class_names, dataset_sizes

# --- Early Stopping 클래스 ---
class EarlyStopping:
    """검증 손실이 개선되지 않을 때 학습을 조기 종료합니다."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """검증 손실이 감소하면 모델을 저장합니다."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, scheduler, early_stopping, num_epochs=25):
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

            # 검증 단계에서 학습률 스케줄러 및 조기 종료 적용
            if phase == 'val':
                scheduler.step(epoch_loss)
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break # 현재 에포크의 훈련/검증 루프 중단
        
        if early_stopping.early_stop:
            break # 전체 에포크 루프 중단

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # 최적 모델 가중치 로드
    model.load_state_dict(torch.load(early_stopping.path))

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
    print(f"Using device: {DEVICE}") # 메시지를 main 함수 안으로 이동
    # 데이터 로드
    dataloaders, class_names, dataset_sizes = load_datasets()
    if dataloaders is None:
        return

    # 모델 구축
    model = build_model(len(class_names))

    # 모델과 샘플 텐서의 디바이스 확인 (CUDA 사용 여부 검증)
    print(f"Model is on: {next(model.parameters()).device}")
    sample_tensor = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(DEVICE)
    print(f"Sample tensor is on: {sample_tensor.device}")

    # 손실 함수와 옵티마이저 정의
    # BCEWithLogitsLoss는 Sigmoid와 BCELoss를 합친 것으로, 수치적으로 안정적임
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 학습률 스케줄러 정의
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=True)

    # 조기 종료 객체 정의
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, delta=MIN_DELTA, path='best_model.pth')

    # 모델 학습
    model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, scheduler, early_stopping, num_epochs=EPOCHS)

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
