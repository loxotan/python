
n, m = map(int, input().split())
grid = [list(map(int, input().split())) for _ in range(n)]
dp = [[-1]*m for _ in range(n)]

import sys
sys.setrecursionlimit(10000)
from collections import deque

def dfs(a, b):
    global cnt
    if (a, b) == (n-1, m-1):
        return 1
    if dp[a][b] != -1:
        return dp[a][b]
    
    ways = 0
    dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for x, y in dir:
        if 0<=a+x<n and 0<=b+y<m and grid[a+x][b+y]<grid[a][b]:
            ways += dfs(a+x, b+y)
    dp[a][b] = ways
    return dp[a][b]

print(dfs(0,0))
