import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)

K = int(input())

def dfs(v):
    global visited
    visited [v] = True
    next = linked[v]
    if not visited[next]:
         dfs(next)

for _ in range(K):
    n = int(input())
    linked = [0] + list(map(int,input().split()))
    visited = [False] * (n+1)
    
    cnt = 0
    for i in range(1, n+1):
        if not visited[i]:
            dfs(i)
            cnt += 1
            
    print(cnt)