import sys
from collections import deque
from itertools import combinations
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]
q = deque([])
res = 1e9

def bfs(): 
    global ans
    while q:
        a, b, d = q.popleft()
        for x, y in [(1,0), (-1,0), (0,1), (0, -1)]:
            if 0<= a+x < n and 0 <= b+y < n and graph[a+x][b+y] != 2:
                q.append([a+x, b+y, d+1]) #주변이 다 비었으면 한번더
            elif 0<= a+x < n and 0 <= b+y < n and graph[a+x][b+y] == 2:
                ans += d+1 #치킨집이 있으면 그만!
                q.clear() # q 비우고
                break #탈출
            
def dist():
    global ans
    a, b, d = q.popleft()
    distance = 1e9
    for o in range(m):
        i, j = case[o]
        distance = min(abs(a-i) + abs(b-j), distance)
    ans += distance

chicken = [] # 전체 치킨집
for i in range(n):
    for j in range(n):
        if graph[i][j] == 2:
            chicken.append([i,j])
            graph[i][j] = 0 # 일단 전부 폐업..

open = [] # 열려있는 치킨집의 경우의 수
for c in combinations(chicken, m):
    open.append(c)
    
for case in open: # 각 경우의 수 마다
    ans = 0
    for l in range(m): # 다시 개업
        i, j = case[l]
        graph[i][j] = 2

    for a in range(n): # 각 집마다 거리 계산
        for b in range(n):
            if graph[a][b] == 1:
                q.append([a,b,0])
                dist()
                
    res = min(res, ans) # 최소의 거리만 기억
    
    for l in range(m): # 다시 폐업
        i, j = case[l]
        graph[i][j] = 0

print(res)