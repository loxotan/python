import sys
input = sys.stdin.readline
from collections import deque

n = int(input())
graph = [[] for _ in range(n+1)]
for _ in range(n-1):
    parent, child, value = map(int, input().split())
    graph[parent].append((child, value))
    graph[child].append((parent, value))

def bfs(start):
    distance = [-1] * (n+1)
    q = deque()
    q.append(start)
    distance[start] = 0
    
    while q:
        now = q.popleft()
        for next, w in graph[now]:
            if distance[next] == -1:
                distance[next] = distance[now]+w
                q.append(next)
    
    max_dist = max(distance)
    farthest_node = distance.index(max_dist)
    return farthest_node, max_dist

u, _ = bfs(1)
v, diameter = bfs(u)
print(diameter)
