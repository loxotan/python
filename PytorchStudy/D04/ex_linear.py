import torch
import torch.nn as nn
## 모델 인스턴스 생성
## 입력 feature : 3개
## 출력 signal : 1개 = 퍼셉트론이 1개
linear_model = nn.Linear(3, 1)

# 모델에 입력 데이터 전달 =>전방향 학습 진행
# 모델 변수명(  텐서 데이터  ) ----> forward() 메서드 실행
data = torch.tensor([1,2,3], dtype=torch.float32) 
output=linear_model(data)
print( output )