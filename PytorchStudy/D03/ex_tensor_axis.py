## ------------------------------------------------------------
## [ Tensor의 축(axis) 변경으로 형태 변환 ]
## 축(axis)
##   1D : shape(n, ) (axis 0, )
##   2D : shape(m, n) (axis 0, axis 1)
##   3D : shape(l, m, n) (axis 0, axis 1, axis 2)
##   4D : shape(k, l, m, n) (axis 0, axis 1, axis 2, axis 3)
## 텐서가 가진 축(axis) 값으로 형태/모양 변경
## ------------------------------------------------------------
## 모듈 로딩
import torch 


## 데이터 텐서 생성
t1=torch.tensor([[11,22], [33,44], [55, 66]])
print(t1.shape, t1.ndim, t1.stride())


## ------------------------------------------------------------
## 축 변경 : 2개 축 변경 
##          3D 경우, 3개 축 중에서 2개만 변경
## ------------------------------------------------------------
## 축 변경 : 2개 축 변경 
## 2D 경우 : 2개 축 
#t1.transpose(1, 0)

# 메서드명 또는 함수명_() : 원본에 적용됨. pandas의 inplace와 동일
t1.transpose_(1, 0)    


## 3D 경우, 3개 축 중에서 2개만 변경
t3=torch.arange(1,13).reshape((1,3,4))
print(t3.shape, t3.ndim, t3.stride())
print(t3)

t31=t3.transpose(0,2)
print(t31.shape, t31.ndim, t31.stride(), t31.is_contiguous())
print(t31)

## ------------------------------------------------------------
## 모든 축 변경 : permute()
## ------------------------------------------------------------ 
dataT=torch.arange(1, 21)
dataT.shape


## 3D ~ 5D 텐서 생성
t3=dataT.reshape((-1, 2, 5))
print(t3.shape, t3.ndim)

t4=dataT.reshape((-1, 1, 2, 5))
print(t4.shape, t4.ndim)

t5=dataT.reshape((-1, 1, 1, 2, 5))
print(t5.shape, t5.ndim)

## 모든 축 변경
t33=t3.permute(2,1,0)
print(t3.shape, t3.ndim)
print(t33.shape, t33.ndim)

t44=t4.permute(3,0,1,2)
print(t4.shape, t4.ndim)
print(t44.shape, t44.ndim)
