## -------------------------------------------------------
## Tensor 생성하기
## -------------------------------------------------------
## 다른 종류의 데이터로 구성된 데이터로 텐서 생성
## -------------------------------------------------------
##- 모듈로딩
import torch
import numpy as np

##- numpy 데이터 생성
data = np.array([1, 'A'])
print(data, data.dtype)

##- 텐서 생성
t0 = torch.tensor([1, 2.3, 0])      # int, float, int ==>float
t1 = torch.tensor([1, 2., 0])       # int, float, int ==>float
#t2 = torch.tensor(data)             # tensor는 수치 데이터만 가능(문자 인식 불가능)
                                    # 수치화 시켜서 저장 (전처리 단계)


##   텐서, 차원,     형태,   데이터 종류
print(t0, t0.ndim, t0.shape, t0.dtype)
print(t1, t1.ndim, t1.shape, t1.dtype)
#print(t2, t2.ndim, t2.shape, t2.dtype)
