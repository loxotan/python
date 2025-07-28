## -------------------------------------------------------
## Tensor 생성하기
## -------------------------------------------------------
## 특정 데이터로 텐서 생성하기
## -------------------------------------------------------
##- 모듈로딩
import torch

##- 텐서 생성
##- [1] 0으로 채워진 텐서 생성 : zeros()
t0 = torch.zeros(2, 3)
t1 = torch.zeros(1, 5, dtype=torch.uint8)


##   텐서, 차원,     형태,   데이터 종류
print(t0, t0.ndim, t0.shape, t0.dtype)
print(t1, t1.ndim, t1.shape, t1.dtype)


##- [2] 1로 채워진 텐서 생성 : ones()
t0 = torch.ones(2, 3)
t1 = torch.ones(2   , 5, 2, dtype=torch.uint8)


##   텐서, 차원,     형태,   데이터 종류
print(t0, t0.ndim, t0.shape, t0.dtype)
print(t1, t1.ndim, t1.shape, t1.dtype)

##- [3] 원하는 값으로 채워진 텐서 생성 : fill()
t0 = torch.fill(t1, 7)
#t1 = torch.fill(1, 5, dtype=torch.uint8)


##   텐서, 차원,     형태,   데이터 종류
print(t0, t0.ndim, t0.shape, t0.dtype)
#print(t1, t1.ndim, t1.shape, t1.dtype)