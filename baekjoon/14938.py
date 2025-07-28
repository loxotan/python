import sys
input = sys.stdin.readline
import heapq

INF = int(10e9)
n, m, r = map(int, input().split())
value = [0] + list(map(int, input().split()))
graph = [[] for _ in range(n+1)]
for _ in range(r):
    a, b, w = map(int, input().split())
    graph[a].append((b, w))
    graph[b].append((a, w))

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

ans = []
for i in range(1, n+1):
    distance = dijkstra(i)
    cnt = 0
    for j in range(1, n+1):
        if distance[j]<=m:
            cnt += value[j]
    ans.append(cnt)

print(max(ans))