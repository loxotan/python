## -------------------------------------------------------
## Tensor 생성하기
## -------------------------------------------------------
## 함수 : tensor(데이터)
## -------------------------------------------------------
##- 모듈로딩
import torch

##- 텐서 생성
##- 데이터 타입 설정 dtype=torch.XXXX
t0 = torch.tensor(100,      dtype=torch.uint8)
t1 = torch.tensor([100],    dtype=torch.float16)
t2 = torch.tensor([[100]],  dtype=torch.bool)

##- 데이터 타입 설정    dtype=torch.XXXX
##- 생성 위치 설정      device=cpu 또는 gpu
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

t0 = torch.tensor(100,      dtype=torch.uint8, device=DEVICE)
t1 = torch.tensor([100],    dtype=torch.float16)
t2 = torch.tensor([[100]],  dtype=torch.bool)

##   텐서, 차원,     형태,   데이터 종류
print(t0, t0.ndim, t0.shape, t0.dtype)
print(t1, t1.ndim, t1.shape, t1.dtype)
print(t2, t2.ndim, t2.shape, t2.dtype)
