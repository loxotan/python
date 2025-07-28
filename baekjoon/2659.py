#ijkl 4자리로 이루어진 숫자의 시계수

from collections import deque

def isSigye(x):
    n = deque()
    for z in range(4):
        n.append(x[z])
    for _ in range(4):
        if ''.join(n) in sigye:
            return False
        n.rotate(1)
    else:
        return True
    
    
sigye = []
for i in range(1, 10):
    for j in range(1, 10):
        for k in range(1, 10):
            for l in range(1, 10):
                num=str(i)+str(j)+str(k)+str(l)
                if isSigye(num):
                    sigye.append(num)

N = deque(map(str, input().split()))
for _ in range(4):
    if ''.join(N) in sigye:
        ans = ''.join(N)
    N.rotate(1)
        
print(sigye.index(ans)+1)
