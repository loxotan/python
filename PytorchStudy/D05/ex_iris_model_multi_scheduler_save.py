## -------------------------------------------------------------
## iris 모델 학습 및 스케줄링과 모델 저장
## -------------------------------------------------------------

## -------------------------------------------------------------
## 모듈 로딩
## -------------------------------------------------------------
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np


## -------------------------------------------------------------
## 데이터 준비 및 전처리
## -------------------------------------------------------------
## - 데이터 로딩
df = pd.read_csv('../Data/iris.csv')

# ## - 데이터 전처리 
df['label'] = LabelEncoder().fit_transform(df['variety'])

X = df[df.columns[:-2]].values.astype('float32')
y = df['label'].values.astype('int64')

# # 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## -------------------------------------------------------------
## 학습, 검증, 테스트 데이터셋 분할
## -------------------------------------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, 
                                                  y, 
                                                  test_size=0.2, 
                                                  random_state=42, 
                                                  stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, 
                                                  y_temp, 
                                                  test_size=0.25, 
                                                  random_state=42, 
                                                  stratify=y_temp)

print(f"Train label distribution: {np.bincount(y_train)}" )
print(f"Val label distribution  : {np.bincount(y_val)}")
print(f"Test label distribution : {np.bincount(y_test)}")

## -------------------------------------------------------------
## 커스텀 모델 정의
## -------------------------------------------------------------
class IrisDataset(Dataset):
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
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)
    
## -------------------------------------------------------------
## 사용자 정의 함수 
## -------------------------------------------------------------
## - 학습 함수 
## -------------------------------------------------------------
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

## -------------------------------------------------------------
## - 검증 함수 
## -------------------------------------------------------------
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

## -------------------------------------------------------------
## - 추론 함수
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


## -------------------------------------------------------------
## 학습 준비 및 진행 
## -------------------------------------------------------------
## ====> (1) 학습 준비
## - 학습 진행 관련 설정값
_LR         = 0.01
_BATCH_SIZE = 16
EPOCHS      = 50

## - 데이터로더 객체 생성
train_loader = DataLoader(IrisDataset(X_train, y_train), batch_size=_BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(IrisDataset(X_val, y_val), batch_size=_BATCH_SIZE)
test_loader  = DataLoader(IrisDataset(X_test, y_test), batch_size=_BATCH_SIZE)

## - 학습 관련 객체 생성 
model        = IrisClassifier()                         ## - 모델 객체 생성
criterion    = nn.CrossEntropyLoss()                    ## - 손실 함수 객체 생성
optimizer    = optim.Adam(model.parameters(), lr=_LR)   ## - 최적화 객체 생성

## - step scheduler - 성능과 관계없이 step size 마다 LR*=gamma
# scheduler    = optim.lr_scheduler.StepLR(optimizer, 
#                                          step_size=20, 
#                                          gamma=0.1)
##- ReduceLROnPlateau - 검증 데이터셋 결과의 성능에 따라
##                      성능이 나쁘면 LR 감소
## mode; max(accuracy 최대화), min(loss 최소화)
scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                 mode='max', factor=0.5, 
                                                 patience=5, verbose=True)

## ====> (2) 학습 진행
## - EPOCH마다 학습 결과 저장 
train_losses, val_losses, val_accuracies = [], [], []

## - 최고 정확도 점수 및 에포크 저장 변수 
best_val_acc    = 0.0  
best_epoch      = 0
stop_cnt        = 5           # LR 감소 횟수 체크 및 조기종료 결정

## - EPOCH만큼 학습 진행
for epoch in range(1, EPOCHS+1):
    ## 학습 후 손실 
    train_loss = train(model, train_loader, criterion, optimizer)
    ## 검증 후 손실 및 성능 
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    ## 학습, 검증 결과 저장 
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)


    ##- 최고 성능 모델 저장
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     ## 모델의 층별 파라미터, 즉 w/b 저장
    #     torch.save(model.state_dict(), f"./best_iris_model.pth")
        
    #     ## 모델 구조 + 층별 파라미터 저장
    #     torch.save(model, "./best_iris_model_all.pt")
    #     print(f"Best model saved at epoch {epoch} with val acc {val_acc:.4f}")

    ##- 최고 성능 모델 저장 + epoch 기록
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        ##- 학습 정보 및 상태 저장, 일반적으로 확장자 .pth (권장)
        torch.save({ 'epoch': best_epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'val_acc': best_val_acc
                    }, 'best_iris_model_state_dict.pth')

        ##- 전체 모델 저장 (구조 + 가중치 포함), 확장자 .pt (권장)
        torch.save(model, 'best_iris_model_entire.pt')

        print(f"Best model saved at epoch {best_epoch} with val acc {best_val_acc:.4f}")

    ##- StepLR경우 학습률 20 epoch마다 10% 감소
    #scheduler.step()  
    ##- ReduceLROnPlateau경우 val_acc를 전달
    scheduler.step(val_acc)
    if scheduler.num_bad_epochs == scheduler.patience:
        print(f'{scheduler.num_bad_epochs} -- {scheduler.patience}')
        stop_cnt -= 1
    
    if stop_cnt == 0:
        print(f'Early stopping at epoch {epoch}, accuracy {val_acc*100:.2f}%...')
        break
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


