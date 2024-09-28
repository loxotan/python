import sys
input = sys.stdin.readline
import heapq

n, e = map(int, input().split())
graph = [[] for _ in range(n+1)]
INF = int(10e9)

for _ in range(e):
    a, b, w = map(int, input().split())
    graph[a].append((b, w))
    graph[b].append((a, w))

v1, v2 = map(int, input().split())

def dijkstra(start):
    distance = [INF]*(n+1)
    distance[start] = 0
    q = []
    heapq.heappush(q, (0, start))
    
    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
            continue
        
        for v, w in graph[now]:
            cost = dist + w
            if cost < distance[v]:
                distance[v] = cost
                heapq.heappush(q, (cost, v))
    
    return distance

distance_from_one = dijkstra(1)
distance_from_v1 = dijkstra(v1)
distance_from_v2 = dijkstra(v2)

#v1 to v2
v1_to_v2 = distance_from_one[v1]+distance_from_v1[v2]+distance_from_v2[n]

#v2 to v1
v2_to_v1 = distance_from_one[v2]+distance_from_v2[v1]+distance_from_v1[n]

result = min(v1_to_v2, v2_to_v1)

if result >= INF:
    print(-1)
else: 
    print(result)
