import sys
input = sys.stdin.readline

N = int(input())
X = [list(map(int, input().split())) for _ in range(N)]
X.sort(key = lambda x:(x[1], x[0]))

ans, end = 0, 0

for i in range(N):
    a, b = X[i]
    if end <= a:
        end = b
        ans += 1
    else:
        continue
    
print(ans)