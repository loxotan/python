import sys
input = sys.stdin.readline
import heapq

n, m, x = map(int, input().split())
INF = int(10e9)
graph = [[] for _ in range(n+1)]

for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))

#갔다가 와야되니까 갈 때 (i -> x) 한번, 올 때 (x -> i) 한번
#각 경우마다 두번 시행되어야 함
def dijkstra(start, end):
    distance = [INF] * (n+1)
    distance[0] = 0
    q = []
    heapq.heappush(q, (0, start))
    distance[start] = 0
    
    while q:
        dist, now = heapq.heappop(q)
        if now == end:
            return dist
        if distance[now] < dist:
            continue
        
        for v, w in graph[now]:
            cost = dist + w
            if cost < distance[v]:
                distance[v] = cost
                heapq.heappush(q, (cost, v))

ans = [0]*(n+1)
for i in range(1, n+1):
    ans[i] = dijkstra(i, x) + dijkstra(x, i)

print(max())