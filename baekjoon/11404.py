import sys
input = sys.stdin.readline

INF = int(1e9)
n = int(input())
m = int(input())

graph = [[INF]*(n+1) for _ in range(n+1)]
for i in range(1, n+1):
    graph[i][i] = 0

for _ in range(m):
    start, end, cost = map(int, input().split())
    if graph[start][end] > cost:
        graph[start][end] = cost

def floyd(graph, n):
    dist = [row[:] for row in graph]
    
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


distance = floyd(graph, n)

for i in range(1, n+1):
    for j in range(1, n+1):
        if distance[i][j] == INF:
            print(0, end = ' ')
        else:
            print(distance[i][j], end = ' ')
    print()