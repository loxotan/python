import sys
input = sys.stdin.readline

K = int(input())

def dfs(v):
    global visited
    visited [v] = True
    for next in linked[v]:
        if not visited[next]:
            dfs(next)

for a in range(K):
    v, e = map(int, input().split())
    linked = [[] for _ in range(v+1)]
    for b in range(e):
        m, n = map(int, input().split())
        linked[m].append(n)
        linked[n].append(m)
    
    visited = [0] * (v+1)
    cnt = 0
    for t in range(1, v+1):
        if not visited[t]:
        
    
    if cnt == 2:
        print('YES')
    else:
        print('NO')



