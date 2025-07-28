import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
arr = []
for _ in range(n):
    arr.append(list(map(int, input().split())))
max = 10e9
ans = [[max]*m for _ in range(n)]

def find_exit(arr, n, m):
    for i in range(n):
        for j in range(m):
            if arr[i][j] == 2:
                return i, j

def search(arr, ans, x, y, n, m):
    pos = [(1,0), (0,1), (-1,0), (0,-1)]
    queue = deque([(x,y)])
    
    while queue:
        x, y = queue.popleft()
        for i, j in pos:
            nx, ny = x+i, y+j
            if 0<=nx<n and 0<=ny<m and arr[nx][ny] == 1 and ans[nx][ny] == max:
                ans[nx][ny] = ans[x][y]+1
                queue.append((nx, ny))

def find_wall(arr, ans, n, m):
    for i in range(n):
        for j in range(m):
            if arr[i][j] == 0:
                ans[i][j] = 0
            elif ans[i][j] == max:
                ans[i][j] = -1

x, y = find_exit(arr, n, m)
ans[x][y] = 0
search(arr, ans, x, y, n, m)
find_wall(arr, ans, n, m)

for row in ans:
    print(*row)


