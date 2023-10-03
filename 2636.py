import sys
from collections import deque
input = sys.stdin.readline

n, m = map(int, input().split())
cheese = [list(map(int, input().split())) for _ in range(n)]

q = deque()

def bfs():
    q.append([0,0])
    while q:
        x, y = q.popleft()
        for xo, yo in ((0,1), (0,-1), (1,0), (-1,0)):
            if 0<=x+xo<n and 0<=y+yo<m:
                if cheese[x+xo][y+yo] == 0:
                    # cheese[x+xo][y+yo] = -1
                    q.append([x+xo,y+yo])
                elif cheese[x+xo][y+yo] == 1:
                    cheese[x+xo][y+yo] = cheese[x][y] + 1
                    q.append([x+xo,y+yo])

bfs()
print(cheese)