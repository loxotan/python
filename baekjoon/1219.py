import sys
input = sys.stdin.readline
from collections import deque

INF = int(1e18)

n, start, end, m = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
city = list(map(int, input().split()))

def bellman(graph, start):
    wallet = [-INF]*(n+1)
    wallet[start] = city[start]
    cycle_nodes = []
    
    for i in range(n):
        for now in range(n):
            if wallet[now] == -INF:
                continue
            for v, w in graph[now]:
                cost = wallet[now]-w+city[v]
                if wallet[v] < cost:
                    wallet[v] = cost
                    if i == n-1:
                        cycle_nodes.append(v)
    
    return wallet, cycle_nodes

wallet, cycle_nodes = bellman(graph, start)

def reach_end(cycle_nodes, end):
    visited = [False]*n
    q = deque(cycle_nodes)

    while q:
        node = q.popleft()
        if node == end:
            return True
        for next in graph[node]:
            v = next[0]
            if not visited[v]:
                visited[v] = True
                q. append(v)
    
    return False
    

if cycle_nodes and reach_end(cycle_nodes, end):
    print('Gee')
else:
    if wallet[end] == -INF:
        print('gg')
    else:
        print(wallet[end])