import sys
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [[False]*(n+1) for x in range(n+1)]
visited = [False]*(n+1)
for _ in range(m):
    a, b = map(int, input().split())
    graph[a][b] = True
    graph[b][a] = True

#아직 True가 되지 못한 visited 마다 dfs 시행

def dfs(v, d):
    global visited
    visited[v] = True
    for next in range(1, n+1):
        if not visited[next] and graph[v][next]:
            return dfs(next, d+1)

cnt = 0
for t in range(1, n+1):
    if not visited[t]:
        dfs(t, 0)
        cnt += 1

print(cnt)

