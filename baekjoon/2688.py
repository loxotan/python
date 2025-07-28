import sys
input = sys.stdin.readline

T = int(input())
for i in range(T):
    n = int(input())
    dp = [[0]*10 for _ in range(n+1)]
    for i in range(10):
        dp[1][i] = 1
    
    for i in range(1, n+1):
        for j in range(10):
            for k in range(j+1):
                dp[i][j] += dp[i-1][k]
    
    print(sum(dp[n]))