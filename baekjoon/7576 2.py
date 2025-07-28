import sys
input = sys.stdin.readline
from collections import deque

m, n = map(int, input().split())
tomato = [[] for _ in range(n)]
for i in range(n):
    tomato[i] = list(map(int, input().split()))
    
#bfs를 한 단계씩 시행해서 동시에 모든 토마토에서 하루씩 익어갈 수 있게
q = deque([])

def bfs():
    while q:
        a, b = q.popleft()
        for io, jo in [(1,0), (-1,0), (0, 1), (0,-1)]:
            if 0<=a+io<n and 0<=b+jo<m and tomato[a+io][b+jo] == 0:
                tomato[a+io][b+jo] = tomato[a][b] + 1 #주변 토마토 익힘
                q.append([a+io, b+jo])

for i in range(n):
    for j in range(m):
        if tomato[i][j] == 1:
            q.append([i,j])

bfs()
res = 0
for i in tomato:
    for j in i:
        if j == 0:
            print(-1)
            exit(0)
    res = max(res, max(i))

print(res-1)    
    