import sys
input = sys.stdin.readline
import heapq

n = int(input())
graph = [[] for _ in range(n+1)]
for _ in range(n):
    info = list(map(int, input().split()))
    v = info[0]
    i = 1
    while True:
        if info[i] == -1:
            break
        graph[v].append((info[i], info[i+1]))
        i+=2

def dijkstra(graph, start):
    distance = [int(1e9)]*(n+1)
    distance[start] = 0
    q = []
    heapq.heappush(q, (0, start))
    
    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
            continue
        
        for v, w in graph[now]:
            cost = dist + w
            if distance[v] > cost:
                distance[v] = cost
                heapq.heappush(q, (cost, v))
    
    max_distance = max(distance[1:])
    farthest = distance.index(max_distance)
    return farthest, max_distance

u, _ = dijkstra(graph, 1)
v, dist = dijkstra(graph, u)

print(dist)

