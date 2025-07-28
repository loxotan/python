import sys
input = sys.stdin.readline
import heapq

INF = int(10e9)
m, n = map(int, input().split())
arr = [[] for _ in range(n)]

for i in range(n):
    map = str(input())
    for j in range(m):
        arr[i].append(int(map[j]))

def search(m, n):
    distance = [[INF]*m for _ in range(n)]
    distance[0][0] = 0
    q = []
    heapq.heappush(q, (0,0,0))
    pos = [(0,1), (1,0), (0,-1), (-1,0)]
    
    while q:
        dist, x, y = heapq.heappop(q)
        if distance[x][y] < dist:
            continue
        
        for dx, dy in pos:
            nx, ny = x+dx, y+dy
            if 0<=nx<n and 0<=ny<m:
                cost = dist+arr[nx][ny]
                if cost<distance[nx][ny]:
                    distance[nx][ny] = cost
                    heapq.heappush(q, (cost, nx, ny))
    
    return distance[n-1][m-1]

print(search(m, n))