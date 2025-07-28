## ---------------------------------------------------------
## Pytorch에서 대량 데이터 처리를 위해서 Dataset/DataLoader생성
## - 데이터셋 클래스/데이터로더 객체 생성
## - 손실이 최저가 되는 데이터의 규칙 및 패턴 찾기
## - 추가 설치 패키지 : conda install scikit-learn
## ---------------------------------------------------------

## ---------------------------------------------------------
## 모듈 로딩
## ---------------------------------------------------------
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

## ---------------------------------------------------------
## 데이터 준비
## ---------------------------------------------------------
## - iris 데이터 로드
data = r'C:\Users\Administrator\Desktop\PytorchStudy\D03\drive-download-20250702T100233Z-1-001\iris.csv'
df = pd.read_csv(data)

## - 데이터 확인
print(df.info())
df.head()

## ---------------------------------------------------------
## 데이터 전처리 
## ---------------------------------------------------------
# 라벨 인코딩 (Setosa, Versicolor, Virginica → 0, 1, 2)
le = LabelEncoder()
df['label'] = le.fit_transform(df['variety'])

# 특성과 라벨 분리 -> df는 tensor 변환 불가, ndarray로 추출
features = df[df.columns[:-2]].values.astype('float32')
labels = df['label'].values.astype('int64')

print(df.head())

# 특성 표준화
scaler = StandardScaler()
features = scaler.fit_transform(features)


## ---------------------------------------------------------
## 커스텀 Dataset 클래스 정의
## ---------------------------------------------------------
class IrisDataset(Dataset):
    def __init__(self, features, labels):
        #self.x = torch.tensor(features, dtype=torch.float32)
        #self.y = torch.tensor(labels, dtype=torch.long)
        self.x = features
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = torch.tensor(self.x[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return data, label

## ---------------------------------------------------------
## 데이터셋 및 데이터로더 생성
## ---------------------------------------------------------
## - 데이터셋 객체 생성
iris_dataset = IrisDataset(features, labels)

## - 데이터로더 객체 생성
iris_loader = DataLoader(iris_dataset, batch_size=16, shuffle=True)

## ---------------------------------------------------------
## 데이터 확인
## ---------------------------------------------------------
## - DataLoader로부터 한 배치의 데이터 가져와서 출력
###### Dataloader는 반복자를 가지고 있음(반복 가능, iteratable)
## - iter() : 이터레이터 객체 생성
## - next() : 다음 배치(batch) 데이터 반환
X_batch, y_batch = next(iter(iris_loader))

print("X_batch shape:", X_batch.shape, X_batch, sep='\n')  # torch.Size([batch_size, feature_dim])
print("y_batch shape:", y_batch.shape, y_batch, sep='\n')  # torch.Size([batch_size])