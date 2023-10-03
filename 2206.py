import sys
from collections import deque
input = sys.stdin.readline

n, m = map(int,input().split())
miro = [[] for _ in range(n)]
for i in range(n):
    line = input()
    for j in range(m):
        miro[i].append(int(line[j]))

#bfs
q = deque()

def bfs(n,m):
    global short
    q.append([0,0,0])
    vst = [[[0]*2 for _ in range(m)]for _ in range(n)]
    vst[0][0][0] = 1
    
    while q:
        i,j,c = q.popleft()
        if i == n-1 and j == m-1:
            short.append(vst[i][j][c])
            return
        
        for x, y in ((1,0), (-1,0), (0,1), (0,-1)):
            if 0<=i+x<n and 0<=j+y<m:
                if vst[i+x][j+y][c] == 0 and miro[i+x][j+y] == 0:
                    q.append([i+x, j+y, c])
                    vst[i+x][j+y][c] = vst[i][j][c] + 1
                elif c == 0 and miro[i+x][j+y] == 1:
                    q.append([i+x, j+y, 1])
                    vst[i+x][j+y][1] = vst[i][j][0] + 1
                    
short = []    
bfs(n, m)

if len(short) == 0:
    print(-1)
else:
    print(min(short))