import sys
from collections import deque
input = sys.stdin.readline

n, m = map(int, input().split())
cheese = [list(map(int, input().split())) for _ in range(n)]


def bfs():
    q = deque()
    q.append([0,0])
    visited = [[False]*m for _ in range(n)]
    visited[0][0] = True
    melt = []
    
    while q:
        x, y = q.popleft()
        cnt = 0
        
        for xo, yo in ((0,1), (0,-1), (1,0), (-1,0)):
            nx, ny = x+xo, y+yo
            if 0<=nx<n and 0<=ny<m and not visited[nx][ny]:
                visited[nx][ny] = True
                if cheese[nx][ny] == 0:
                    q.append([nx,ny])
                else:
                    melt.append([nx, ny])
    
    for x, y in melt:
        cheese[x][y] = 0
    
    return len(melt)

ans = 0
while True:
    remaining_cheese = sum(sum(row) for row in cheese)
    if remaining_cheese == 0:
        break
    cnt = bfs()
    ans += 1
    last_cheese = cnt

print(ans)
print(last_cheese)