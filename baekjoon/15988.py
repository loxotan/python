import sys
input = sys.stdin.readline

dp = [[0,0,0] for _ in range(1_000_001)]
dp[1] = [1,0,0]
dp[2] = [1,1,0]
dp[3] = [2,1,1]
for i in range(4, 1_000_001):
    dp[i][0] = (dp[i-1][0]+dp[i-1][1]+dp[i-1][2])%1_000_000_009
    dp[i][1] = (dp[i-2][0]+dp[i-2][1]+dp[i-2][2])%1_000_000_009
    dp[i][2] = (dp[i-3][0]+dp[i-3][1]+dp[i-3][2])%1_000_000_009
    
T = int(input())
for _ in range(T):
    N = int(input())
    print(sum(dp[N])%1_000_000_009)