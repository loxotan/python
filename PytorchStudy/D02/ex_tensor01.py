## -------------------------------------------------------
## Tensor 생성하기
## -------------------------------------------------------
## 함수 : tensor(데이터)
## -------------------------------------------------------
##- 모듈로딩
import torch

##- 텐서 생성
t0 = torch.tensor(100)
t1 = torch.tensor([100])
t2 = torch.tensor([[100]])

##   텐서, 차원,     형태,   데이터 종류
print(t0, t0.ndim, t0.shape, t0.dtype)
print(t1, t1.ndim, t1.shape, t1.dtype)
print(t2, t2.ndim, t2.shape, t2.dtype)
