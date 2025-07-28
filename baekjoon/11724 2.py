import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)

def dfs(v, d):
    global visited
    visited[v] = True
    for i in graph[v]:
        if not visited[i]:
            dfs(i, d+1)
            
n, m = map(int, input().split())
graph = [[] for _ in range(n+1)]
for i in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)
    
cnt = 0
visited = [False]*(n+1)
for i in range(1, n+1):
    if not visited[i]:
        dfs(i, 1)
        cnt += 1
        
print(cnt)