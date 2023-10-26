import sys
import copy
input = sys.stdin.readline

n = int(input())
rgb = [0]*(n+1)
INF = 10e9
ans = INF
for i in range(n):
    rgb[i] = list(map(int, input().split()))

for j in range(3):
    dp = [[INF,INF,INF] for _ in range(n)]   
    dp [0][j] = rgb[0][j] 
    for i in range(1, n):
        dp[i][0] = min(dp[i-1][1], dp[i-1][2]) + rgb[i][0]
        dp[i][1] = min(dp[i-1][0], dp[i-1][2]) + rgb[i][1]
        dp[i][2] = min(dp[i-1][0], dp[i-1][1]) + rgb[i][2]

    for k in range(3):
        if j != k:
            ans = min(ans, dp[-1][k])
            
print(ans)
