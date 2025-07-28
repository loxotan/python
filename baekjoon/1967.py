#1. 트리의 리프 노드 찾기
#2. 리프노드마다 가장 멀리 있는 노드 찾기
#3. 그 중 가장 큰 값 출력

import sys
input = sys.stdin.readline
import heapq

INF = int(1e9)
n = int(input())
graph = [[] for _ in range(n+1)]
for _ in range(n-1):
    parent, child, value = map(int, input().split())
    graph[parent].append((child, value))


#1
def children_node():
    children = []
    
    def traverse(node):
        if not graph[node]:
            children.append(node)
        else:
            for child, value in graph[node]:
                traverse(child)
    
    traverse(1)
    return children

#2
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
    
    return distance[1:]

#3
children = children_node()

for i in range(1, n+1):
    if graph[i]:
        for child, value in graph[i]:
            if (i, value) in graph[child]:
                continue
            else:
                graph[child].append((i, value))

diameter = []
for child in children:
    diameter.append(max(dijkstra(child)))

print(max(diameter))