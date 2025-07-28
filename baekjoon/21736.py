import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
arr = []
for i in range(n):
    row = input().rstrip()
    arr.append([0]*m)
    for j in range(m):
        if row[j] == 'O':
            arr[i][j] = 0
        elif row[j] == 'I':
            arr[i][j] = 2
        elif row[j] == 'X':
            arr[i][j] = 3
        else: #'P'
            arr[i][j] = 1

def find_doyeon(arr, n, m):
    for i in range(n):
        for j in range(m):
            if arr[i][j] == 2:
                return i, j

def search(arr, i, j, n, m):
    visited = [[False]*m for _ in range(n)]
    visited[i][j] = True
    q = deque([(i,j)])
    pos = [(1,0), (0,1), (-1,0), (0,-1)]
    ans = 0
    while q:
        x, y = q.popleft()
        for dx, dy in pos:
            nx, ny = x+dx, y+dy
            if 0<=nx<n and 0<=ny<m and not visited[nx][ny] and arr[nx][ny]<3:
                if arr[nx][ny] == 1:
                    ans += 1
                visited[nx][ny] = True
                q.append((nx,ny))
    return ans

x, y = find_doyeon(arr, n, m)
ans = search(arr, x, y, n, m)

if ans == 0:
    print('TT')
else:
    print(ans)
