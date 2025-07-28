import sys
input = sys.stdin.readline

n = int(input())
podoju = [int(input()) for _ in range(n)]

dp = [[0] for _ in range(n+1)]
dp[0] = podoju[0]
if n>1:
    dp[1] = podoju[0]+podoju[1]
if n>2:
    dp[2] = max(podoju[2]+podoju[0], podoju[2]+podoju[1], dp[1])
    for i in range(3, n):
        dp[i] = max(podoju[i]+dp[i-2], podoju[i]+podoju[i-1]+dp[i-3], dp[i-1])

print(dp[n-1])
