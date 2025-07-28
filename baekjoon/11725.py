import sys
input = sys.stdin.readline
from collections import deque

n = int(input())
graph = [[] for _ in range(n+1)]

for _ in range(n-1):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

def search(graph, n):
    q = deque([1])
    visited = [False]*(n+1)
    visited[1] = True
    parent = [0]*(n+1)
        
    while q:
        current_node = q.popleft()
        for i in graph[current_node]:
            if not visited[i]:
                q.append(i)
                parent[i] = current_node
                visited[i] = True

    return parent

parent_list = search(graph, n)
for i in range(2, n+1):
    print(parent_list[i])