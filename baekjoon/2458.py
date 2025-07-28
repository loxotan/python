import sys
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [[False]*(n+1) for _ in range(n+1)]

for _ in range(m):
    short, tall = map(int, input().split())
    graph[short][tall] = True

def floyd(graph, n):
    reachable = [row[:] for row in graph]
    
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                if reachable[i][k] and reachable[k][j]:
                    reachable[i][j] = True
    
    return reachable

reach = floyd(graph, n)
ans = 0
for i in range(1, n+1):
    cnt = 0
    for j in range(1, n+1):
        if reach[i][j] or reach[j][i]:
            cnt += 1
    if cnt == n-1:
        ans += 1

print(ans)