import sys
input = sys.stdin.readline

N, K = map(int, input().split())
X = list(map(int, input().split()))

ans = 0
for i in range(1, N):
    if X[i] <= X[i-1] + K:
        continue
    else:
        ans += 1
        
print(ans)