import sys
input = sys.stdin.readline

# 오른쪽 1 <> 3
# 위쪽 2 <> 4
# 왼쪽 3 <> 1
# 아래쪽 4 <> 2

n = int(input())
original = list(map(int, input().split()))
answer = []
from collections import deque

def check_same(shape, original):
    shape = deque(shape)
    for j in range(n):
        if list(shape) == original or list(shape)[::-1] == original:
            return True
        else:
            shape.rotate(-1)

    return False

m = int(input())
for _ in range(m):
    
    shape = list(map(int, input().split()))
    clockwise = check_same(shape, original)
    
    other_shape = [0]*(n)
    for i in range(n):
        other_shape[i] = (shape[i]+2)%4
        if other_shape[i] == 0:
            other_shape[i] = 4
    counter_clockwise = check_same(other_shape, original)
    
    if clockwise or counter_clockwise:
        answer.append(shape)

print(len(answer))
for ans in answer:
    print(*ans)