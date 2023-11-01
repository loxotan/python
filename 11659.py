import sys
input = sys.stdin.readline

N, M = map(int, input().split())
X = list(map(int, input().split()))
plus = [0]
for i in range(1, N+1):
    plus.append(plus[i-1] + X[i-1])
    
for _ in range(M):
    a, b = map(int, input().split())
    print(plus[b] - plus[a-1])