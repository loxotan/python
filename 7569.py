import sys
input = sys.stdin.readline
from collections import deque

m, n, h = map(int, input().split())
tomato = [[[] for _ in range(n)] for _ in range(h)]
for i in range(h):
    for j in range(n):
        tomato[i][j] = list(map(int, input().split()))

#bfs를 한 단계씩 시행해서 동시에 모든 토마토에서 하루씩 익어갈 수 있게
q = deque([])

def bfs():
    while q:
        c, a, b = q.popleft()
        for z, x, y in [(0,1,0), (0,-1,0), (0,0, 1), (0,0,-1),(1,0,0),(-1,0,0)]:
            if 0<=a+x<n and 0<=b+y<m and 0<=c+z<h and tomato[c+z][a+x][b+y] == 0:
                tomato[c+z][a+x][b+y] = tomato[c][a][b] + 1 #주변 토마토 익힘
                q.append([c+z, a+x, b+y])
                
for z in range(h):
    for i in range(n):
        for j in range(m):
            if tomato[z][i][j] == 1:
                q.append([z,i,j])

bfs()
res = 0

for i in tomato:
    for j in i:
        for k in j:
            if k == 0:
                print(-1)
                exit(0)
        res = max(res, max(j))

print(res-1)    
    