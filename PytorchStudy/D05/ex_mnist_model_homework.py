## -------------------------------------------------------------
## 목표 : iris 꽃의 3개 품종 식별하는 모델 개발
## -------------------------------------------------------------
## - 조사  결과 : 품종에 따라 꽃잎의 길이/너비, 꽃받침의 길이/너비가 다름
## - 데이터수집 : iris.csv
## - 데이터분석 : 수집된 데이터로 품종 3개 구분 가능 여부 분석

## - 데이터  셋 : 학습용, 검증용, 테스트용 준비
## - 모델  개발 : 분류 모델 => 분류 알고리즘 선택  => 학습 진행
## - 모델  검사 : 3개 구분 성능 체크 후 재학습 또는 종료 결정
## -------------------------------------------------------------

## -------------------------------------------------------------
## 모듈 로딩
## -------------------------------------------------------------
## -pytorch에서 데이터 및 모델 관련 
import torch                                    #텐서 및 수치함수
import torch.nn as nn                           #신경망관련
import torch.optim as optim                     #모델 최적화 관련
from torch.utils.data import Dataset, DataLoader#데이터셋 & 로더

## -데이터 로딩 및 전처리 관련
import pandas as pd                                             #데이터 로딩/분석
from sklearn.preprocessing import LabelEncoder, StandardScaler  #피처/타겟 전처리
from sklearn.model_selection import train_test_split            #학습/검증/테스트 분리
import numpy as np                                              #raw 데이터 관련

## -모델 성능 평가 및 시각화 모듈
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #성능 평가 관련
import matplotlib.pyplot as plt                                      #시각화


## -------------------------------------------------------------
## 데이터 준비 및 전처리
## -------------------------------------------------------------
## - 데이터 로딩
data_train = pd.read_csv('../Data/mnist_train.csv', header=None)  # iris.csv 파일 경로에 맞게 수정
data_test = pd.read_csv('../Data/mnist_test.csv', header=None)    # iris.csv 파일 경로에 맞게 수정

## - 데이터 전처리
train_X = data_train[data_train.columns[1:]].values.astype('float32')
train_Y = data_train[data_train.columns[0]].values.astype('int64')

test_X = data_test[data_test.columns[1:]].values.astype('float32')
test_Y = data_test[data_test.columns[0]].values.astype('int64')

## 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)
# X_scaled = scaler.fit_transform(X)

## -------------------------------------------------------------
## 학습, 검증, 테스트 데이터셋 분할
## -------------------------------------------------------------

X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, 
                                                    train_Y, 
                                                    test_size=0.25, 
                                                    random_state=42, 
                                                    stratify=train_Y)

print(f"Train label distribution: {np.bincount(Y_train)}" )
print(f"Val label distribution  : {np.bincount(Y_val)}")
print(f"Test label distribution : {np.bincount(test_Y)}")

## -------------------------------------------------------------
## 커스텀 모델 정의
## -------------------------------------------------------------
class MNISTDataset(Dataset):
    def __init__(self, features, labels):
        self.x = features
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feature = torch.tensor(self.x[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return feature, label   

## -------------------------------------------------------------
## 커스텀 클래스 모델 정의
## 주의!! 다중분류라면 Softmax() 활성함수를 사용해야 하지만,
## 손실함수를 CrossEntropyLoss로 잡으면 손실함수 내에서 자동으로 Softmax() 처리
## 활성함수에서 Softmax()를 쓰면 2번 쓰이게 되므로 
## 활성함수를 Softmax()를 사용하지 않아야 함
## 다른 손실함수를 쓸 때에는 손실함수 내부의 function을 뜯어볼 필요 
## -------------------------------------------------------------   
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential( #시퀀셜의 장점: 자동으로 전달해줌
            nn.Linear(28*28, 256),     # = 포워드할 때 편함
            nn.ReLU(),
            nn.Linear(256, 32),    # 입력값 4개, 은닉층 16개, 은닉층 32개
            nn.ReLU(),
            nn.Linear(32, 10)      ### 회귀분석을 하겠습니다.
        )                         #   -> 회귀/이진/다중에 따라 & 손실에 따라

    def forward(self, x):
        return self.net(x)
    
## -------------------------------------------------------------
## 사용자 정의 함수 
## -------------------------------------------------------------
## - 학습 함수 
## 학습할 때는 역전파가 필요
## 검증할 때는 역전파 끄기
## -------------------------------------------------------------
def train(model, dataloader, criterion, optimizer):
    ## 모델 모드 설정 - 학습
    model.train()

    ## 에포크 단위 손실 저장 변수
    total_loss = 0

    ## 배치 사이즈 만큼 학습 진행
    for x_batch, y_batch in dataloader:
        ## 옵티마이저 경사하강법
        optimizer.zero_grad()               # weight/bias 초기화
        outputs = model(x_batch)            # 순방향 학습
        loss = criterion(outputs, y_batch)  # 손실 계산
        loss.backward()                     # 새로운 weight/bias 계산(역전파)
        optimizer.step()                    # 새로운 weight/bias 업데이트
        total_loss += loss.item()           # 배치 단위의 손실 저장(누적)
    return total_loss / len(dataloader)     # 평균 배치 단위 손실값

## -------------------------------------------------------------
## - 검증 함수 
## - 한 에포크 끝날 때 마다 모델 상태 체크
## - 현재 weight/bias로 검증용 데이터를 추론/평가
## - !!! weight/bias가 업데이트 되면 안됨 !!!
## -------------------------------------------------------------
def evaluate(model, dataloader, criterion):
    ## 모델 모드 설정 - 검증 : weight/bias 업데이트 X, autograd X
    model.eval()

    ## 검증 결과 저장 변수: 손실, 정답 개수
    total_loss = 0
    correct = 0
    total = 0

    ## weight/bias 업데이트 안되도록 기능 off(no grad)
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)                    # 검증 시작

            loss = criterion(outputs, y_batch)          # 손실 계산(얼마나 틀렸나요?)
            total_loss += loss.item()                   

            preds = outputs.argmax(dim=1)               # 예측값 추출(무엇이 가장 확률이 높은지?)
            
            correct += (preds == y_batch).sum().item()  # 정답 = 예측값인 것만 correct에 추가
            total += y_batch.size(0)                    # 총 문제 수
    accuracy = correct / total                          # 정확도
    return total_loss / len(dataloader), accuracy

## -------------------------------------------------------------
## - 추론 함수
## - 변수를 넣으면 예측값을 알려준다
## - 모델이 완성되면 서비스에 활용되는 함수
## - " 이거는 무엇 입니다. "
## -------------------------------------------------------------
def predict(model, data):

    ## 모델 동작 모드 : 검증/추론
    model.eval()

    with torch.no_grad():
        if isinstance(data, torch.Tensor):
            x_tensor = data.clone().detach()                    # 텐서라면 복사해서 사용
        else:
            x_tensor = torch.tensor(data, dtype=torch.float32)  # 텐서가 아니면 텐서로 만들기

        outputs = model(x_tensor)                               # 예측 (확률값)
        predictions = outputs.argmax(dim=1)                     # 정수 라벨 추출(가장 그럴싸한 대답 고르기)
    return predictions


## -------------------------------------------------------------
## 학습 준비 및 진행 
## -------------------------------------------------------------
## ====> (1) 학습 준비
## - 학습 진행 관련 설정값: 러닝레이트, 배치사이즈, 에포크 수
_LR         = 0.01
_BATCH_SIZE = 16
EPOCHS      = 50

## - 데이터로더 객체 생성: 학습, 검증, 추론
train_loader = DataLoader(MNISTDataset(X_train, Y_train), batch_size=_BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(MNISTDataset(X_val, Y_val), batch_size=_BATCH_SIZE)
test_loader  = DataLoader(MNISTDataset(test_X_scaled, test_Y), batch_size=_BATCH_SIZE)

## - 학습 관련 객체 생성: 모델, 손실함수, 최적화
model        = MNISTClassifier()                         ## - 모델 객체 생성
criterion    = nn.CrossEntropyLoss()                    ## - 손실 함수 객체 생성
optimizer    = optim.Adam(model.parameters(), lr=_LR)   ## - 최적화 객체 생성
#                            -> 모델이 가지고 있는 인풋/아웃풋 넘겨주기

## ====> (2) 학습 진행
## - EPOCH마다 학습 결과 저장 
train_losses    = []
val_losses      = []
val_accuracies  = []

## - EPOCH만큼 학습 진행
for epoch in range(1, EPOCHS+1):
    ## 학습 후 손실 
    train_loss = train(model, train_loader, criterion, optimizer)
    ## 검증 후 손실 및 성능 
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    ## 학습, 검증 결과 저장 
    train_losses.append(train_loss)
    val_losses.append(val_loss)         # 진행 될수록 낮아져야
    val_accuracies.append(val_acc)      # 진행 될수록 높아져야

    ## 10회 마다 출력
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


## -------------------------------------------------------------
## 테스트 진행 
## ------------------------------------------------------------- 
for x_batch, y_batch in test_loader:
    y_pre = predict(model, x_batch)
    print(y_pre)

## - 테스트 셋에 대한 검증 결과 추출 
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


## - 전체 예측값과 실제값 수집
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_targets.extend(y_batch.numpy())

# Confusion Matrix 생성 및 시각화
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Test Set")
plt.show()

# -------------------------------------------------------------
# 학습 곡선 시각화
# -------------------------------------------------------------
## - Loss 그래프
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()


## - Accuracy  그래프
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------
# 새로운 샘플 추론
# -------------------------------------------------------------
# new_samples = np.array([
#     [5.1, 3.5, 1.4, 0.2],
#     [6.7, 3.0, 5.2, 2.3],
#     [5.9, 3.0, 4.2, 1.5],
# ], dtype='float32')

# ## - 학습용 데이터셋과 동일한 전처리 진행
# ## - 스케일링 적용
# new_samples_scaled = scaler.transform(new_samples)

# ## - 예측
# pred_labels = predict(model, new_samples_scaled)
# label_names = df['variety'].unique()

# print("\n[ New Sample Prediction ]")
# for i, pred in enumerate(pred_labels):
#     print(f"Sample {i+1}: Predicted Class = {label_names[pred]}")

# 다음시간 : 
# 변화가 없으면 중단(조기종료)
# 모델 저장하기
# 저장한 모델 불러와서 사용하기
# mnist 분류하는 모델 만들어보기