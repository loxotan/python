import sys
input = sys.stdin.readline

N = int(input())
grid = [list(input().rstrip()) for _ in range(N)]

def check(a, b):
    ans = 0
    for i in [a, a+1]:
        cnt = 1
        for j in range(1, N):
            if i<=N-1 and grid[i][j] == grid[i][j-1]:
                cnt += 1
            else:
                ans = max(ans, cnt)
                cnt = 1
        ans = max(ans, cnt)
    
    for j in [b, b+1]:
        cnt = 1
        for i in range(1, N):
            if j<=N-1 and grid[i][j] == grid[i-1][j]:
                cnt += 1
            else:
                ans = max(ans, cnt)
                cnt = 1
        ans = max(ans, cnt)
    return ans
        
ans = 0

for i in range(N):
    for j in range(N):
        ans = max(ans, check(i,j))
        if j+1<=N-1:
            grid[i][j], grid[i][j+1] = grid[i][j+1], grid[i][j]
            ans = max(ans, check(i, j))
            grid[i][j], grid[i][j+1] = grid[i][j+1], grid[i][j]
        if i+1<=N-1:
            grid[i][j], grid[i+1][j] = grid[i+1][j], grid[i][j]
            ans = max(ans, check(i, j))
            grid[i][j], grid[i+1][j] = grid[i+1][j], grid[i][j]
        
print(ans)