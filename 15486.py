import sys
input = sys.stdin.readline

N = int(input())
schedule = [list(map(int, input().split())) for _ in range(N)]
dp = [0 for i in range(N+1)]

w=0
for i in range(N):
    w = max(dp[i], w)
    if schedule[i][0] <= N-i:
        dp[i+schedule[i][0]] = max(w+schedule[i][1], dp[i+schedule[i][0]])
        

print(max(dp))