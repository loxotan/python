import sys
input = sys.stdin.readline
from collections import deque

INF = int(1e18)

n, m = map(int, input().split())
graph = [[] for _ in range(n+1)]
rev_graph = [[] for _ in range(n+1)]
for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    rev_graph[v].append((u, w))


def bellman(graph, start):
    wallet = [-INF]*(n+1)
    wallet[start] = 0
    predecessor = [None]*(n+1)
    infinite_nodes = set()
    
    for i in range(n):
        for now in range(1, n+1):
            if wallet[now] == -INF:
                continue
            for v, w in graph[now]:
                cost = wallet[now]+w
                if wallet[v] < cost:
                    wallet[v] = cost
                    predecessor[v] = now
                    if i == n-1:
                        infinite_nodes.add(v)
    
    return wallet, infinite_nodes, predecessor


def reconstruct_path(parent, start, end):
    path = []
    node = end
    while node != start:
        path.append(node)
        node = parent[node]
        if node is None:
            return None
    path.append(start)
    path.reverse()
    
    return path


def bfs(start, graph):
    visited = [False]*(n+1)
    visited[start] = True
    q = deque()
    q.append(start)
    
    while q:
        u = q.popleft()
        for v, w in graph[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    
    return visited


def main():
    wallet, infinite_nodes, order = bellman(graph, 1)
    if wallet[n] == -INF:
        return print(-1)
    
    visited_from_start = bfs(1, graph)
    visited_to_end = bfs(n, rev_graph)
    infinite = False
    for node in infinite_nodes:
        if visited_from_start[node] and visited_to_end[node]:
            infinite = True
            break
    
    if infinite:
        return print(-1)
    else:
        path = reconstruct_path(order, 1, n)
        return print(*path)

main()