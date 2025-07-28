import sys
from collections import deque
input = sys.stdin.readline

m, n, k = map(int, input().split())
jongee = [[0]*n for _ in range(m)]
for T in range(k):
    x1, y1, x2, y2 = map(int, input().split())
    for j in range(x1, x2):
        for i in range(y1, y2):
            jongee[i][j] = 1

vst = [[False]*n for _ in range(m)]
q = deque()

def bfs(a, b):
    vst[a][b] = True
    q.append([a,b])
    cnt = 1
    while q:
        a, b = q.popleft()
        for c, d in ((1,0), (-1,0), (0,1), (0,-1)):
            if 0<=a+c<m and 0<=b+d<n and not vst[a+c][b+d] and jongee[a+c][b+d] == 0:
                q.append([a+c,b+d])
                vst[a+c][b+d] = True
                cnt += 1
    nurbi.append(cnt)

nurbi = []
for i in range(m):
    for j in range(n):
        if not vst[i][j] and jongee[i][j] == 0:
            bfs(i,j)

print(len(nurbi))
nurbi.sort()
print(*nurbi)