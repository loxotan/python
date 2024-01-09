import sys
input = sys.stdin.readline

T = int(input())

dp = [[0,0,0], [1,0,0], [0,1,0], [1,1,1]] + [[0,0,0] for _ in range(99_999)]

# [마지막에 +1, 마지막에 +2, 마지막에 +3]
for i in range(4, 100_001):
        dp[i][0] = (dp[i-1][1] + dp[i-1][2])%1_000_000_009
        dp[i][1] = (dp[i-2][0] + dp[i-2][2])%1_000_000_009
        dp[i][2] = (dp[i-3][0] + dp[i-3][1])%1_000_000_009
    
for _ in range(T):
    N = int(input())
    print(sum(dp[N])%1_000_000_009)
    