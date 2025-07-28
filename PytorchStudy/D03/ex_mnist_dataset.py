## -------------------------------------------------------------
## 목표 : mnist digit handwritten number 식별하는 모델 개발
## - 조사  결과 : 0 ~ 9까지 손으로 작성된 이미지 
## - 데이터수집 : mnist_train.csv,  mnist_test.csv
## - 데이터분석 : 학습용, 테스트용 
## - 모델  개발 : 분류 모델 => 분류 알고리즘 선택  => 학습 진행
## -------------------------------------------------------------

## -------------------------------------------------------------
## 모듈 로딩
## -------------------------------------------------------------
import torch
import pandas as pd                                   ##- 데이터 전처리 및 분석용 모듈
from sklearn.model_selection import train_test_split  ##- 데이터셋 분리 함수 
from sklearn.preprocessing import LabelEncoder        ##- 전처리 모듈
from torch.utils.data import Dataset, DataLoader

## -------------------------------------------------------------
## 데이터 로딩
## -------------------------------------------------------------
## - 함수 read_csv('파일경로/파일명') : csv 파일을 표 형태의 DataFrame
##                                   에 저장하는 함수
## -------------------------------------------------------------
##- 데이터 로딩
DATA_TRAIN_FILE = r'C:\Users\Administrator\Desktop\PytorchStudy\D03\drive-download-20250702T100233Z-1-001\mnist_train.csv'
DATA_TEST_FILE  = r'C:\Users\Administrator\Desktop\PytorchStudy\D03\drive-download-20250702T100233Z-1-001\mnist_test.csv'

minstDF=pd.read_csv(DATA_TEST_FILE, header=None)

## -------------------------------------------------------------
## 피쳐와 타겟 분리
## -------------------------------------------------------------
##- 피쳐 : 이미지 픽셀 값
featureDF = minstDF.iloc[:, minstDF.columns[1:]] ## 0자리 이후, 784개

##- 타겟 : 이미지가 나타내는 숫자
targetDF = minstDF.iloc[:, minstDF.columns[0]] ## 0자리

##- 피쳐와 타겟 체크
print(f'피쳐 : {featureDF.shape}  타겟 : {targetDF.shape}')

## -------------------------------------------------------------
## 데이터 전처리 => 피쳐 : 정규화
## -------------------------------------------------------------
##- 피쳐 즉, 픽셀값 0.0 ~ 1.0
featureDF = featureDF / 255.
print(featureDF.head(3))

## -------------------------------------------------------------
## 데이터 셋 분리
## -------------------------------------------------------------
## - 학습용, 테스트용 분리 => 80:20
## - 분류 모델인 경우 주의점 : 분류 종류마다의 데이터가 3개 데이터셋에 모두
##                          존재 해야함
x_train, x_test, y_train, y_test = train_test_split(featureDF, 
                                                    targetDF,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=targetDF)

## - 학습용, 검증용 분리
x_train, x_val, y_train, y_val   = train_test_split(x_train, 
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y_train)

## 체크
import numpy as np

_indexs = ['Train', 'Val', 'Test']

rateDF=pd.DataFrame([np.bincount(y_train),np.bincount(y_val), np.bincount(y_test)], index=_indexs )
print(rateDF)
## 개수가 even한지 체크!!

## -------------------------------------------------------------
## 커스텀 데이터셋 클래스 정의 
## -------------------------------------------------------------
## 클래스이름 : MnistDataset
## 부모클래스 : Dataset
## 매개  변수 : featureDF, targetDF
## 오버라이딩 : 3개 메서드
##          - def _ _init_ _(self, featureDF, targetDF)
##          - def _ _len_ _(self)
##          - def _ _getitem_ _(self, index)
## -------------------------------------------------------------
class MnistDataset(Dataset):

    # - 피쳐와 타겟 그리고 추가 저장 데이터 정의
    def __init__(self, dataDF, labelDF):
        super().__init__()
        self.data  = dataDF.values          # ndarray 피쳐 [필수]
        self.label = labelDF.values         # ndarray 타겟 [필수]
        self.length = dataDF.shape[0]       # (행, 열)
        self.data_name = dataDF.columns     ### 필요한 정보가 있다면 추가!

    # - 데이터세의 데이터 갯수 반환 
    def __len__(self):
        return self.length

    # - 특정 인덱스의 피쳐와 타겟 반환
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), torch.LongTensor(self.label[index])

## 데이터 준비 완료!