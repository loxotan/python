import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)
from collections import deque

K = int(input())

def dfs(v):
    global visited, result
    if result == False:
        return result
    
    elif visited[v] == 1:
        for next in linked[v]:
            if visited[next] == 0:
                visited[next] = -1
                dfs(next)
            elif visited[next] == 1:
                result = False
                
    elif visited[v] == -1:
        for next in linked[v]:
            if visited[next] == 0:
                visited[next] = 1
                dfs(next)
            elif visited[next] == -1:
                result = False
    

for a in range(K):
    v, e = map(int, input().split())
    linked = [[] for _ in range(v+1)]
    for b in range(e):
        m, n = map(int, input().split())
        linked[m].append(n)
        linked[n].append(m)
    
    visited = [0] * (v+1)
    result = True
    visited[1] = 1
    dfs(1)
    
    if result == True:
        print('YES')
    else:
        print('NO')



