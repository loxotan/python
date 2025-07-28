import sys
input = sys.stdin.readline
from collections import deque

r, c = map(int, input().split())
cave = []
for _ in range(r):
    cave.append(list(input().strip()))
n = int(input())
sticks = list(map(int, input().split()))

def find_floating(cave):
    total_mineral = set()
    grounded_mineral = set()
    q = deque()
    visited = [[False]*c for _ in range(r)]
    pos = [(1,0), (0,1), (-1,0), (0,-1)]
    
    # 전체 미네랄 수 확인
    for i in range(r):
        for j in range(c):
            if cave[i][j] == 'x':
                if i == r-1:
                    visited[i][j] = True
                    grounded_mineral.add((i,j))
                    q.append((i,j))
                total_mineral.add((i,j))
    
    while q:
        x, y = q.popleft()
        
        for dx, dy in pos:
            nx, ny = x+dx, y+dy
            if 0<=nx<r and 0<=ny<c and not visited[nx][ny] and cave[nx][ny] == 'x':
                visited[nx][ny] = True
                grounded_mineral.add((nx, ny))
                q.append((nx,ny))
    
    floating_mineral = list(total_mineral - grounded_mineral)
    return floating_mineral
    
def gravity(cluster):
    fall = r
    cluster = sorted(cluster, key = lambda x:(x[1], -x[0]))
    check = []
    for mineral in cluster:
        x, y = mineral
        if y not in check:
            for i in range(x+1, r):
                if i == r-1 and cave[i][y] == '.': # 바닥까지 떨어짐
                    fall = min(fall, i-x)
                elif cave[i][y] == 'x':            # 중간에 부딛힘
                    fall = min(fall, i-x-1)
            check.append(y)
    if fall == r:
        fall = 0

    return fall


def move_cluster(cave, cluster, fall):
    for x, y in cluster:
        cave[x][y] = '.'
    for x, y in cluster:
        cave[x+fall][y] = 'x'


def main():
    for t in range(n):
        throw = sticks[t]
        row = cave[r-throw]
        if t%2 == 0:   #왼쪽에서 날아오는 막대기
            for i in range(c):
                if row[i] == 'x':
                    row[i] = '.'
                    break
        elif t%2 == 1: #오른쪽에서 날아오는 막대기
            for i in range(c-1, -1, -1):
                if row[i] == 'x':
                    row[i] = '.'
                    break
        
        cluster = find_floating(cave)
        if cluster:
            fall = gravity(cluster)
            if fall > 0:
                move_cluster(cave, cluster, fall)
    
    for row in cave:
        print(''.join(row))

main()