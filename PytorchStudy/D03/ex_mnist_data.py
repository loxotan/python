## -------------------------------------------------------------
## 목표 : mnist digit handwritten number 식별하는 모델 개발
## - 조사  결과 : 0 ~ 9까지 손으로 작성된 이미지 , 28x28 
## - 데이터수집 : mnist_train.csv,  mnist_test.csv
## - 데이터분석 : 학습용, 테스트용 
## - 모델  개발 : 분류 모델 => 분류 알고리즘 선택  => 학습 진행
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
DATA_TRAIN_FILE = r'C:\Users\Administrator\Desktop\PytorchStudy\D03\drive-download-20250702T100233Z-1-001\mnist_train.csv'
DATA_TEST_FILE  = r'C:\Users\Administrator\Desktop\PytorchStudy\D03\drive-download-20250702T100233Z-1-001\mnist_test.csv'

minstDF=pd.read_csv(DATA_TEST_FILE, header=None)

##- 데이터 확인 :  info(), head()
minstDF.info()           ## 데이터에 대한 컬럼별 정보 요약 출력
print(minstDF.head())    ## DataFrame에 저장된 데이터 출력

## -------------------------------------------------------------
## 피쳐와 타겟 분리
## -------------------------------------------------------------
##- 피쳐 : 이미지 픽셀 값
featureDF = minstDF.iloc[:, 1:] #0자리부터 끝까지

##- 타겟 : 이미지가 나타내는 숫자
targetDF = minstDF.iloc[:, 0] #0자리

##- 피쳐와 타겟 체크
print(f'피쳐 : {featureDF.shape}  타겟 : {targetDF.shape}')


## -------------------------------------------------------------
## 숫자 손글씨 이미지와 라벨 체크 
## -------------------------------------------------------------

# 20개 2줄에 시각화 
nrow, ncol = 4, 5
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 12))

axes = axes.flatten()

for idx in range(nrow * ncol):
    ## - 이미지 데이터 1D ==> 2D (28, 28)
    imgData = featureDF.iloc[idx].values.reshape(-1, 28)

    ## - 이미지 그리기
    axes[idx].imshow(imgData, cmap='gray')
    axes[idx].set_title(f'No.{targetDF[idx]}')

    ## - x축, y축 눈금 숨기기
    #axes[idx].axis('off')
    axes[idx].xaxis.set_visible(False)
    axes[idx].yaxis.set_visible(False)

plt.tight_layout(h_pad=3., w_pad=3.)
plt.show()

