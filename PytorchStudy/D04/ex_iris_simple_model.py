## -------------------------------------------------------------
## 목표 : iris 꽃의 3개 품종 식별하는 모델 개발
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
import pandas as pd                                   ##- 데이터 전처리 및 분석용 모듈
from sklearn.model_selection import train_test_split  ##- 데이터셋 분리 함수 
from sklearn.preprocessing import LabelEncoder        ##- 전처리 모듈

import torch                                          ##- 텐서 및 기본 함수 모듈
import torch.nn as nn                                 ##- 인공신경망 관련 모듈

## -------------------------------------------------------------
## 데이터 로딩
## -------------------------------------------------------------
## - 함수 read_csv('파일경로/파일명') : csv 파일을 표 형태의 DataFrame
##                                   에 저장하는 함수
## -------------------------------------------------------------
##- 데이터 로딩
irisDF=pd.read_csv('../Data/iris.csv')

## -------------------------------------------------------------
## 데이터 전처리 : 학습이 가능한 상태로 만들기 
## -------------------------------------------------------------
## - 품종 컬럼을 숫자화 
irisDF['label']=LabelEncoder().fit_transform(irisDF['variety'])


## -------------------------------------------------------------
## 피쳐와 타겟 분리
## -------------------------------------------------------------
feature = irisDF.loc[:, irisDF.columns[:-2]]
target  = irisDF[irisDF.columns[-1]]

print(f'feature : {feature.shape}   target : {target.shape}')
print(feature.head(), target.head(), sep='\n')


## -------------------------------------------------------------
## 데이터 셋 분리
## -------------------------------------------------------------
## - 학습용, 테스트용 분리 => 80:20
## - 분류 모델인 경우 주의점 : 분류 종류마다의 데이터가 3개 데이터셋에 모두
##                          존재 해야함
##   예) 학습 : setosa, Virginica             테스트 : Versicolor   X
##       학습 : setosa, Virginica,Versicolor  테스트 : setosa, Virginica,Versicolor
x_train, x_test, y_train, y_test = train_test_split(feature, 
                                                    target,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=target)

## - 학습용, 검증용 분리
x_train, x_val, y_train, y_val   = train_test_split(x_train, 
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y_train)

## 체크
import numpy as np
print(f"Train label : {np.bincount(y_train)}" )
print(f"Val label   : {np.bincount(y_val)}" )
print(f"Test label  : {np.bincount(y_test)}" )

## -------------------------------------------------------------
## 모델 구성      입력        출력        활성함수
## - 입력층 :     4           10         ReLU    <= 피쳐 수  입력
## - 은닉층 :     10          5          ReLU 
## - 출력층 :     5           3          Softmax <= 분류 개수 출력
## -------------------------------------------------------------
irisModel=nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3), 
        nn.Softmax(dim=1)       
)
# 이진분류 : A일 확률 vs B일 확률( = 1 - A일 확률)
#            -> 출력을 A일 확률만 뽑아도 됨( = 출력층 1개)
# 다중분류 : A와 나머지 vs B와 나머지 vs C와 나머지
#            -> 각각 이진분류( = 출력층 3개)
#            -> A일 확률 + B일 확률 + C일 확률 != 1.0
#                 -> = 1.0으로 만들어주는 함수 softmax

print(f'irisModel => \n{irisModel}')

## -------------------------------------------------------------
## 학습 진행 
## - 모델 입력과 출력 타입 : Tensor
## -------------------------------------------------------------
## DataFrame ==> ndarry ==> Tensor
x_train_ts = torch.tensor(x_train.values, dtype=torch.float32)
y_train_ts = torch.tensor(y_train.values, dtype=torch.long)


## 학습 => 모델객체변수명(피쳐데이터)
output=irisModel(x_train_ts)

## 모델 예측 확률값 중에서 가장 큰 확률값의 레이블 
y_pred=output.argmax(dim=1)

print(y_train_ts.shape,  y_pred.shape)

## 정답과 예측값 체크 
print( (y_train_ts == y_pred) )
print( (y_train_ts == y_pred).sum().item() )