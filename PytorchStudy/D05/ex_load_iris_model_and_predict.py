## -------------------------------------------------------------
## 저장 모델 활용 및 적용
## -------------------------------------------------------------

## -------------------------------------------------------------
## 모듈 로딩
## -------------------------------------------------------------
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from iris_classes import *


## -------------------------------------------------------------
## 전체 모델 불러오기
## ------------------------------------------------------------- 
# model = torch.load('./best_iris_model_entire.pt', weights_only=False)
# model.eval()

## -------------------------------------------------------------
## 모델 파라미터 불러오기
## ------------------------------------------------------------- 
model = IrisClassifier()
checkpoint = torch.load('./best_iris_model_state_dict.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

## -------------------------------------------------------------
## 입력 샘플 준비
## ------------------------------------------------------------- 
samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3]
], dtype='float32')


## -------------------------------------------------------------
## 학습용 데이터와 동일한 전처리 
## ------------------------------------------------------------- 
##- iris.csv 로부터 scaler 재구성
df = pd.read_csv('../Data/iris.csv')
X = df[df.columns[:-1]].values.astype('float32')
scaler = StandardScaler()
scaler.fit(X)
samples_scaled = scaler.transform(samples)

## -------------------------------------------------------------
## 예측
## -------------------------------------------------------------
with torch.no_grad():
    inputs = torch.tensor(samples_scaled, dtype=torch.float32)
    outputs = model(inputs)
    preds = outputs.argmax(dim=1)
    print(f"예측 결과: {preds}")