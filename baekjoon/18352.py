import sys
input = sys.stdin.readline
import heapq

INF = int(10e9)
n, m, k, x = map(int, input().split())
graph = [[] for _ in range(n+1)]

for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)

def dijkstra(start):
    distance = [INF]*(n+1)
    distance[start] = 0
    q = []
    heapq.heappush(q, (0, start))
    
    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
            continue
        
        for v in graph[now]:
            cost = dist + 1
            if cost < distance[v]:
                distance[v] = cost
                heapq.heappush(q, (cost, v))
    
    return distance

distance = dijkstra(x)
ans = []
for i in range(1, n+1):
    if distance[i] == k:
        ans.append(i)

if len(ans)==0:
    print(-1)
else:
    for an in ans:
        print(an)