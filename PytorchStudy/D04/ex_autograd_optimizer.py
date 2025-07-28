## ---------------------------------------------------
## Optimizer 최적화 
## - 최적의 성능을 내기 위한 기울기값 계산 및 업데이트 진행
## - torch.optim.XXX : 최적화 객체 서브 모듈
## - optim.XXX()     : 다양한 방식의 기울기 업데이트 객체
## - .zero_grad()    : .grad 속성 값을 초기화 메서드
## - .step()         : .grad 속성 깂 업데이트 메서드   
## ---------------------------------------------------
## ---------------------------------------------------
## 모듈 로딩
## ---------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

## ---------------------------------------------------
## 업데이트 간격 설정 변수 선언
## ---------------------------------------------------
# 학습률
_lr = 0.1

## ---------------------------------------------------
## Tensor 생성 
## ---------------------------------------------------
## 초기 값 설정 (학습할 파라미터 w)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

## 최적화 객체 
optimizer = torch.optim.SGD([w, b], lr=_lr)

## 입력과 정답 정의 (y = 2x)
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

# 로깅용 리스트
epochs, w_list, b_list, loss_list = [], [], [], []


## ---------------------------------------------------
## Tensor 값 자동 업데이트 
## ---------------------------------------------------
print("-" * 52)
print(f"{'Epoch':^5} | {'w':^8} | {'b':^8} | {'Loss':^10} | {'grad':^10}")
print("-" * 52)


for epoch in range(1, 11):
    # 순전파 및 예측 
    y_pred = w * x + b

    # 손실: MSE
    loss = ((y_pred - y_true) ** 2).mean()

    # 역전파 및 파라미터 업데이트
    optimizer.zero_grad()  # 그래디언트 초기화
    loss.backward()        # 기울기 계산
    optimizer.step()       # 파라미터 업데이트

    # 기록
    epochs.append(epoch)
    w_list.append(w.item())
    b_list.append(b.item())
    loss_list.append(loss.item())

    # 출력
    #print(f"Epoch {epoch:2d}: Loss = {loss.item():.4f} | w = {w.item():.4f} | b = {b.item():.4f}")
    print(f"{epoch:^5} | {w.item():^8.4f} | {b.item():^8.4f} | {loss.item():^10.6f} | {w.grad.item():^10.6f}")