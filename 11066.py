from collections import deque

T = int(input())
for _ in range(T):
    n = int(input())
    X = list(map(int, input().split()))
    X.sort()
    
    ans = 0
    while len(X) != 1:
        a = X.pop(0)
        b = X.pop(0)
        X.append(a+b)
        ans += (a+b)
        X.sort()
        
    print(ans)