import sys
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [[] for _ in range(n+1)]
for _ in range(m):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))

def bellman(graph, start):
    distance = [int(1e9)] * (n+1)
    distance[start] = 0
    negative = False
    
    for i in range(n):
        for now in range(1, n+1):
            if distance[now] != int(1e9):
                for next, weight in graph[now]:
                    if distance[now] + weight < distance[next]:
                        distance[next] = distance[now] + weight
                        if i == n-1:
                            negative = True
    
    return distance, negative

dist, neg = bellman(graph, 1)
if neg:
    print(-1)
else:
    for i in range(2, len(dist)):
        if dist[i] >= int(1e9):
            print(-1)
        else:
            print(dist[i])