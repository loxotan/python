import sys
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    n = int(input())
    
    d = [0]*(n+1)
    d[0] = 0
    d[1] = 1
    if n>1:
        d[2] = 1
    if n>2:
        d[3] = 1
    if n>3:
        d[4] = 2
    if n>4:
        d[5] = 2
    if n>5:
        for i in range(6, n+1):
            d[i] = d[i-1]+d[i-5]
            
    print(d[n])