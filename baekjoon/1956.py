import sys
input = sys.stdin.readline

#floyd
INF = int(1e9)
n, m = map(int, input().split())
graph = [[INF]*(n+1) for _ in range(n+1)]

for _ in range(m):
    a, b, c = map(int, input().split())
    graph[a][b] = c

def floyd(graph, n):
    dist = [row[:] for row in graph]
    
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

dist = floyd(graph, n)
ans = []
for i in range(1, n+1):
    ans.append(dist[i][i])
min_dist = min(ans)
if min_dist >= INF:
    print(-1)
else:
    print(min_dist)