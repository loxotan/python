import sys
input = sys.stdin.readline

T = int(input())
graph = [list(map(int, input().split())) for _ in range(T)]

dp = [[[0]*T for _ in range(T)] for _ in range(3)]
dp[0][0][1] = 1

for i in range(T):
    for j in range(T):
        if graph[i][j] == 1:
            continue

        # 가로: (i, j-1)에서 넘어옴
        if j-1 >= 0:
            dp[0][i][j] += dp[0][i][j-1] + dp[2][i][j-1]

        # 세로: (i-1, j)에서 넘어옴
        if i-1 >= 0:
            dp[1][i][j] += dp[1][i-1][j] + dp[2][i-1][j]

        # 대각: (i-1, j-1)에서 넘어옴 + 3칸 비어야 함
        if i-1 >= 0 and j-1 >= 0 and graph[i-1][j] == 0 and graph[i][j-1] == 0:
            dp[2][i][j] += dp[0][i-1][j-1] + dp[1][i-1][j-1] + dp[2][i-1][j-1]

print(dp[0][T-1][T-1] + dp[1][T-1][T-1] + dp[2][T-1][T-1])