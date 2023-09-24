import sys
input = sys.stdin.readline

K = int(input())

def dfs(v):
    global visited
    visited [v] = True
    for next in linked[v]:
        if not visited[next]:
            dfs(next)
            