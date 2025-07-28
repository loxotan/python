## -------------------------------------------------------------
## 목표 : MNIST DIGIT HANDWRITTEN 식별하는 모델 개발
## -------------------------------------------------------------
## - 조사  결과 : 28x28 크기의 숫자 손글씨 이미지 데이터, 흑백
## - 데이터수집 : mnist_train.csv, mnist_test.csv
## - 데이터분석 : 0 ~ 9 즉, 10개 숫자 이미지 분류 
##
## - 데이터  셋 : 학습용, 검증용, 테스트용 준비
## - 모델  개발 : 분류 모델 => 분류 알고리즘 선택  => 학습 진행
##               인공신경망 알고리즘 => DNN 모델
## -------------------------------------------------------------

## -------------------------------------------------------------
## 모듈 로딩
## -------------------------------------------------------------
## - 데이터 분석 및 로딩
import pandas as pd    

## - 텐서, 인공신경망, 최적화 관련 모듈
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## - 데이터 전처리 및 분활용 모듈
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

## - 데이터셋과 데이터로더 관련 모듈
from torch.utils.data import Dataset, DataLoader
import numpy as np

## - 학습결과 시각화 및 평가 지표
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


## -------------------------------------------------------------
## 데이터 준비
## -------------------------------------------------------------
##- 데이터 로딩
DATA_TRAIN_FILE = '../data/mnist_train.csv'
DATA_TEST_FILE  = '../data/mnist_test.csv'

##- csv 파일 첫번째 줄이 컬럼명 아님 => header=None
trainDF= pd.read_csv(DATA_TRAIN_FILE, header=None)
testDF = pd.read_csv(DATA_TEST_FILE, header=None)


## -------------------------------------------------------------
## 피쳐와 타겟 분리
## -------------------------------------------------------------
##- 피쳐 : 이미지 픽셀 값
X_train_all = trainDF.iloc[:, 1:] / 255.0
X_test      = testDF.iloc[:, 1:]  / 255.0

##- 타겟 : 이미지가 나타내는 숫자
y_train_all = trainDF.iloc[:, 0 ]
y_test      = testDF.iloc[:, 0 ]

##- 피쳐와 타겟 체크
print(f'[TRAIN ALL] 피쳐 : {X_train_all.shape}  타겟 : {y_train_all.shape}')
print(f'[   TEST  ] 피쳐 : {X_test.shape}  타겟 : {y_test.shape}')
## -------------------------------------------------------------
## 전처리 => 피쳐 정규화 
## -------------------------------------------------------------
## - 픽셀값 0 ~ 255 ==> 0.0 ~ 1.0  : 학습 안정성 및 모델 일반화 
featureDF = X_train_all/255.
print(featureDF.head(3))

## -------------------------------------------------------------
## 학습, 검증, 테스트 데이터셋 분할
## -------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(  X_train_all, 
                                                    y_train_all, 
                                                    test_size=0.25, 
                                                    random_state=42, 
                                                    stratify=y_train_all)


print(f"Train label distribution: {np.bincount(y_train)}" )
print(f"Val label distribution  : {np.bincount(y_val)}")

## -------------------------------------------------------------
## 커스텀 모델 정의
## -------------------------------------------------------------
class MnistDataset(Dataset):
    def __init__(self, features, labels):
        self.x = features
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        featrue = torch.tensor(self.x[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return featrue, label   

## -------------------------------------------------------------
## 커스텀 클래스 모델 정의
## -------------------------------------------------------------   
class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)
    
## -------------------------------------------------------------
## 사용자 정의 함수 
## -------------------------------------------------------------
## - 함수기능 : 1에포크 학습 후 loss 반환
## - 함수이름 : train
## - 매개변수 : model, dataloader, criterion, optimizer
## - 함수결과 : loss 반환
## -------------------------------------------------------------
def train(model, dataloader, criterion, optimizer):
    #- 모델 동작 모드 설정 : 학습모드
    model.train()

    #- 배치크기만큼 전방향 학습 & 역전파 
    total_loss = 0    # 1에포크(처음부터 끝까지 학습)에 대한 전체 손실저장
    for x_batch, y_batch in dataloader:
        #- 전방향(forward) 학습
        optimizer.zero_grad()
        outputs = model(x_batch)

        #- 예측값과 정답 차이 계산 즉, 손실/비용함수 처리
        loss = criterion(outputs, y_batch)

        #- 손실값을 미분 진행 
        loss.backward()
        #- 모델의 W, b에 새로운 값 업데이트
        optimizer.step()
        #- 배치크기만큼 데이터에 대한 손실 추가 저장
        total_loss += loss.item()

    return total_loss / len(dataloader)

## -------------------------------------------------------------
## - 함수기능 : 현재 모델의 W, b로 검증데이터에 대한 검증 후 loss 반환
## - 함수이름 : evaluate
## - 매개변수 : model, dataloader, criterion
## - 함수결과 : loss, 정확도 반환
## -------------------------------------------------------------
def evaluate(model, dataloader, criterion):
    ## - 모델 동작 모드 설정 : 검증 모드
    model.eval()

    ## - 검증데이터셋에 대한 손실, 정확도 저장 변수
    total_loss = 0
    correct = 0
    total = 0

    ## - 검증데이터셋으로 검증 진행
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            ##- 예측값 추출
            outputs = model(x_batch)

            ##- 예측값과 정답 차이 계산 
            loss = criterion(outputs, y_batch)

            ##- 차이 즉, 손실 누적
            total_loss += loss.item()

            ##- 예측 타겟 추출
            preds = outputs.argmax(dim=1)
            ##- 정답 타겟과 예측 타겟 비교 및 카운팅
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

## -------------------------------------------------------------
## - 함수기능 : 데이터에 대한 예측 반환
## - 함수이름 : evaluate
## - 매개변수 : model, dataloader, criterion
## - 함수결과 : loss, 정확도 반환
## -------------------------------------------------------------
def predict(model, data):
    model.eval()
    with torch.no_grad():
        if isinstance(data, torch.Tensor):
            x_tensor = data.clone().detach()
        else:
            x_tensor = torch.tensor(data, dtype=torch.float32)

        outputs = model(x_tensor)
        predictions = outputs.argmax(dim=1)
    return predictions


## --------------------------------------------------------------------
## 학습 준비
## --------------------------------------------------------------------
## - 학습 진행 관련 설정값
_LR         = 0.01
_BATCH_SIZE = 16
EPOCHS      = 51
step_cnt = 5

## - 데이터로더 객체 생성 
train_loader = DataLoader(MnistDataset(X_train.values, y_train.values), batch_size=_BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(MnistDataset(X_val.values, y_val.values), batch_size=_BATCH_SIZE)
test_loader  = DataLoader(MnistDataset(X_test.values, y_test.values), batch_size=_BATCH_SIZE)

## - 학습관련 객체 생성
model        = MnistClassifier()                      ## - 모델 객체 생성
criterion    = nn.CrossEntropyLoss()                  ## - 손실 함수 객체 생성
optimizer    = optim.Adam(model.parameters(), lr=_LR) ## - 최적화 객체 생성 
scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=5)  ## - 학습률 스케줄러
## - 학습결과 저장 변수들
train_losses, val_losses, val_accuracies = [], [], []


## --------------------------------------------------------------------
## 학습 진행
## --------------------------------------------------------------------
for epoch in range(1, EPOCHS):
    ## 학습 후 손실 
    train_loss = train(model, train_loader, criterion, optimizer)

    ## 검증 후 손실 및 성능 
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    ## Epoch단위 학습/검증 결과 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    scheduler.step(val_acc)
    
    if scheduler.num_bad_epochs >= scheduler.patience:
        step_cnt -= 1
        print(f'step count remaining: {step_cnt}, learning rate reduced')
    
    if step_cnt == 0:
        print(f'stopping early at epoch {epoch} due to no improvement')
        break
    
    ## 10회 마다 출력
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


## -------------------------------------------------------------
## 테스트 평가 및 Confusion Matrix
## -------------------------------------------------------------
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Accuracy: {test_acc:.4f}")


## ===> Confusion Matrix
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch).argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# -------------------------------------------------------------
# 학습 곡선 시각화
# -------------------------------------------------------------
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.legend()
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# -------------------------------------------------------------
# 새로운 샘플 예측 테스트
# -------------------------------------------------------------
sample = X_test.iloc[[0]].values
pred = predict(model, sample)
print(f"예측 결과: {pred.item()}, 실제 정답: {y_test.iloc[0]}")


# -------------------------------------------------------------
# Accuracy는 맞췄지만 Loss가 높은 샘플 추출 예시
# -------------------------------------------------------------
# 예측 결과와 정답 비교 (MNIST 기준)
def find_high_loss_correct_predictions(model, dataloader, criterion, num_samples=5):
    model.eval()
    results = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1)
            loss_per_sample = F.cross_entropy(outputs, y_batch, reduction='none')

            for i in range(len(y_batch)):
                if preds[i] == y_batch[i]:
                    results.append((loss_per_sample[i].item(), preds[i].item(), y_batch[i].item(), x_batch[i]))

    results.sort(reverse=True, key=lambda x: x[0])  # loss 기준 내림차순
    print(f"📌 Accuracy는 맞았지만 Loss가 높은 샘플 {num_samples}개")
    for i in range(min(num_samples, len(results))):
        loss, pred, true, _ = results[i]
        print(f"Sample {i+1}: Loss={loss:.4f}, 예측={pred}, 정답={true}")


find_high_loss_correct_predictions(model, val_loader, criterion)
