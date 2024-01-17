import sys
input = sys.stdin.readline

cnt = 1

while True:
    N = int(input())
    if N == 0:
        break
    
    grid = [list(map(int, input().split())) for _ in range(N)]
    dp = [[0,0,0] for _ in range(N)]
    INF = 1_000_000
    dp[0] = [INF, grid[0][1], grid[0][2]+grid[0][1]]
    for i in range(1, N):
        dp[i][0] = min(dp[i-1][0], dp[i-1][1]) + grid[i][0]
        dp[i][1] = min(dp[i][0], dp[i-1][0], dp[i-1][1], dp[i-1][2]) + grid[i][1]
        dp[i][2] = min(dp[i][1], dp[i-1][1], dp[i-1][2]) + grid[i][2]
        
    print(cnt, '. ', dp[N-1][1], sep = '')
    cnt += 1