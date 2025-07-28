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

## -------------------------------------------------------------
## 데이터 로딩
## -------------------------------------------------------------
## - 함수 read_csv('파일경로/파일명') : csv 파일을 표 형태의 DataFrame
##                                   에 저장하는 함수
## -------------------------------------------------------------
##- 데이터 로딩
irisDF=pd.read_csv(r'C:\Users\Administrator\Desktop\PytorchStudy\D03\drive-download-20250702T100233Z-1-001\iris.csv')

## -------------------------------------------------------------
## 데이터 전처리 : 학습이 가능한 상태로 만들기 
## -------------------------------------------------------------
## - 품종 컬럼을 숫자화 
print('고유값' , irisDF['variety'].unique() )

## -'Setosa' 'Versicolor' 'Virginica' => 0, 1, 2
# irisDF['variety'].replace('Setosa', 0, inplace=True)
# irisDF['variety'].replace('Versicolor', 1, inplace=True)
# irisDF['variety'].replace('Virginica', 2, inplace=True)
irisDF['label']=LabelEncoder().fit_transform(irisDF['variety'])

print(irisDF.head())
irisDF.info()

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
##   예) 학습 : Setosa, Virginica              테스트 : Versicolor   X
##       학습 : Setosa, Virginica, Versicolor  테스트 : Setosa, Virginica, Versicolor
x_train, x_test, y_train, y_test = train_test_split(feature, 
                                                    target,
                                                    test_size=0.2,# = 0.8은 train_size
                                                    random_state=42,
                                                    stratify=target)# = 원본 'target' 비율에 맞춰서 쪼개기

## - 학습용, 검증용 분리
x_train, x_val, y_train, y_val   = train_test_split(x_train, # 학습용 데이터에서 쪼개기
                                                    y_train,
                                                    test_size=0.2, # 학습용 데이터의 20% 테스트
                                                    random_state=42,
                                                    stratify=y_train)

## 체크
import numpy as np
print(f"Train label : {np.bincount(y_train)}" )
print(f"Val label   : {np.bincount(y_val)}" )
print(f"Test label  : {np.bincount(y_test)}" )