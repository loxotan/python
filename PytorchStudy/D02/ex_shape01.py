## -------------------------------------------------------
## Tensor 형태 변경하기
## -------------------------------------------------------
##- 모듈로딩
import torch

##- 텐서 생성
t1 = torch.tensor([1,3,5,7,9,11])

print(f't1 => {t1.shape}, {t1.ndim}D')

##1D => 2D
t2 = t1.unsqueeze(dim=0)
print(f't2 => {t2.shape}, {t2.ndim}D')

t2 = t1.unsqueeze(dim=1)
print(f't2 => {t2.shape}, {t2.ndim}D')

## 2D => 3D
t3 = t2.unsqueeze(dim=1)
print(f't3 => {t3.shape}. {t3.ndim}D')

## 3D => 1D
t11 = t3.squeeze()
print(f't11 => {t11.shape}, {t11.ndim}D')

## 3D => 2D
t22 = t3.squeeze(dim=1)
print(f't22 => {t22.shape}, {t22.ndim}D')

t22 = t3.squeeze(dim=0)
print(f't22 => {t22.shape}, {t22.ndim}D')