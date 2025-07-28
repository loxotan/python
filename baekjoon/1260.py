import sys
input = sys.stdin.readline

def DFS(idx):
    global visited
    visited[idx] = True
    print(idx, end = ' ')
    for next in range(1, n+1):
        if not visited[next] and graph[idx][next]:
            DFS(next)
            
def BFS():
    global q, visited
    while q:
        cur = q.pop(0)
        print(cur, end = ' ')
        for next in range(1, n+1):
            if not visited[next] and graph[cur][next]:
                visited[next] = True
                q.append(next)            
                
n, m, v = map(int, input().split())
graph = [[False] * (n+1) for _ in range(n+1)]
#간선 넣기
for _ in range(m):
    a, b = map(int, input().split())
    graph[a][b] = True
    graph[b][a] = True
#DFS
visited = [False]*(n+1)     
DFS(v)
print()
#BFS
visited = [False]*(n+1)
q = [v]
visited[v] = True
BFS()
    
