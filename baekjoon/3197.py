import sys
input = sys.stdin.readline
from collections import deque

r, c = map(int, input().split())
lake = []
ducks = []
water_q = deque()

for i in range(r):
    row = list(input().strip())
    for j in range(c):
        if row[j] == 'L':
            ducks.append((i,j))
            water_q.append((i,j))
            row[j] = '.'
        elif row[j] == '.':
            water_q.append((i,j))
    lake.append(row)

pos = [(1,0), (0,1), (-1,0), (0,-1)]


def main(water_q):
    day = 0
    duck_q = deque()
    duck_q.append(ducks[0])
    visited_ducks = [[False]*c for _ in range(r)]
    visited_ducks[ducks[0][0]][ducks[0][1]] = True
    
    visited_water = [[False]*c for _ in range(r)]
    for x, y in water_q:
        visited_water[x][y] = True
    
    while True:
        duck_q_temp = deque()
        while duck_q:
            x, y = duck_q.popleft()
            if (x, y) == ducks[1]:
                return day
            
            for dx, dy in pos:
                nx, ny = x+dx, y+dy
                if 0<=nx<r and 0<=ny<c and not visited_ducks[nx][ny]:
                    visited_ducks[nx][ny] = True
                    if lake[nx][ny] == '.':
                        duck_q.append((nx, ny))
                    elif lake[nx][ny] == 'X':
                        duck_q_temp.append((nx, ny))
        duck_q = duck_q_temp
        
        water_q_temp = deque()
        while water_q:
            x, y = water_q.popleft()
            
            for dx, dy in pos:
                nx, ny = x+dx, y+dy
                if 0<=nx<r and 0<=ny<c and not visited_water[nx][ny]:
                    if lake[nx][ny] == 'X':
                        lake[nx][ny] = '.'
                        visited_water[nx][ny] = True
                        water_q_temp.append((nx, ny))
        water_q = water_q_temp

        day += 1
    
print(main(water_q))