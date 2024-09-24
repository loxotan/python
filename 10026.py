import sys
input = sys.stdin.readline
from collections import deque

n = int(input())
arr = [input().rstrip() for _ in range(n)]
colorblind = [row.replace('G', 'R') for row in arr]

def count(arr, n):
    visited = [[False]*n for _ in range(n)]
    ans = 0
    pos = [(1,0),  (0,1), (-1,0), (0,-1)]
    for i in range(n):
        for j in range(n):
            if not visited[i][j]:
                q = deque([(i, j)])
                visited[i][j] = True
                color = arr[i][j]
                
                while q:
                    x, y = q.popleft()
                    
                    for di, dj in pos:
                        ni, nj = x+di, y+dj
                        if 0<=ni<n and 0<=nj<n and not visited[ni][nj] and arr[ni][nj] == color:
                            visited[ni][nj] = True
                            q.append((ni, nj))
                ans += 1
    return ans

print(count(arr, n), count(colorblind, n))
