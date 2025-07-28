## -------------------------------------------------------------
## 목표 : iris 꽃의 3개 품종 식별하는 모델 개발
## - 조사  결과 : 품종에 따라 꽃잎의 길이/너비, 꽃받침의 길이/너비가 다름
## - 데이터수집 : iris.csv
## - 데이터분석 : 수집된 데이터로 품종 3개 구분 가능 여부 분석
## - 모델  개발 : 분류 모델 => 분류 알고리즘 선택  => 학습 진행
## - 모델  검사 : 3개 구분 성능 체크 후 재학습 또는 종료 결정
## -------------------------------------------------------------

## -------------------------------------------------------------
## 모듈 로딩
## -------------------------------------------------------------
import pandas as pd                 ##- 데이터 전처리 및 분석용 모듈
import matplotlib.pyplot as plt     ##- 데이터 시각화 모듈

## -------------------------------------------------------------
## 데이터 로딩
## -------------------------------------------------------------
## - 함수 read_csv('파일경로/파일명') : csv 파일을 표 형태의 DataFrame
##                                   에 저장하는 함수
## -------------------------------------------------------------
##- 데이터 로딩
#irisDF=pd.read_csv(r'C:\Users\Administrator\Desktop\PytorchStudy\D03\drive-download-20250702T100233Z-1-001\iris.csv')
irisDF = pd.read_csv('./iris.csv')
##- 데이터 확인 :  info(), head()
#irisDF.info()           ## 데이터에 대한 컬럼별 정보 요약 출력
#print(irisDF.head())    ## DataFrame에 저장된 데이터 출력

## -------------------------------------------------------------
## 피쳐와 타겟 분리
## -------------------------------------------------------------
##- 피쳐 : 품종을 식별하는 속성으로 꽃받침 길이/너비, 꽃잎 길이/너비 컬럼
print(f'컬럼/열명 : {irisDF.columns}')
print(f'행       : {irisDF.index}')

featureDF = irisDF.loc[:, irisDF.columns[:-1]]

##- 타겟 : 품종명 => 숫자로 변경 
targetDF = irisDF.loc[:, irisDF.columns[-1]]

##- 피쳐와 타겟 체크
print(f'피쳐 : {featureDF.shape}')
print(f'타겟 : {targetDF.shape}')

## -------------------------------------------------------------
## 품종별 분류 가능여부 시각화 
## -------------------------------------------------------------
## 꽃받침 피쳐 2개와 꽃잎 피쳐 2개로 타겟 분류 시각화 
fig, axes = plt.subplots(1,2, figsize=(12, 5))

axes = axes.flatten()

labels   = ['Setosa', 'Versicolor', 'Virginica']
colors   = ['red', 'green', 'orange']
colnames = [['sepal.length', 'sepal.width'], ['petal.length', 'petal.width']]
rowIdx   = [50, 100, 150]

for idx, (col1, col2) in zip(range(2), colnames):
    for _label, color, rows in zip( labels, colors, rowIdx):
        axes[idx].scatter(irisDF.loc[(rows-50):rows, col1], 
                    irisDF.loc[(rows-50):rows, col2], 
                    c=color, label=_label)
    axes[idx].set_title(f"[{col1} & {col2}")
    axes[idx].legend()
plt.show()
