import sys
input = sys.stdin.readline
import heapq

n = int(input())
m = int(input())

INF = int(10e9)
graph = [[] for i in range(n+1)]
distance = [INF]*(n+1)

for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))

start, end = map(int, input().split())

def dijkstra(start):
    q = []
    heapq.heappush(q, (0, start))
    distance[start] = 0
    
    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
            continue
        
        for v, w in graph[now]:
            cost = dist+w
            if cost < distance[v]:
                distance[v] = cost
                heapq.heappush(q, (cost, v))

dijkstra(start)
print(distance[end])
