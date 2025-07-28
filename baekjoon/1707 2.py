import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)
from collections import deque

K = int(input())

def dfs(v):
    global visited, result
    next = deque()
    next.append(v)
    while next:
        node = next.pop()
        if visited[node] == 1: #1로 색칠된 노드에서
            for n in linked[node]:
                if visited[n] == 1: #다음 노드가 1로 색칠되면
                    result = False #이분그래프아님
                elif visited[n] == 0: #다음 노드가 색칠안되어 있으면
                    next.append(n) #한번더
                    visited[n] = -1#그리고 -1로 색칠
        
        elif visited[node] == -1:
            for n in linked[node]:
                if visited[n] == -1:
                    result = False
                elif visited[n] == 0:
                    next.append(n)
                    visited[n] = 1
                
for a in range(K):
    v, e = map(int, input().split())
    linked = [[] for _ in range(v+1)]
    for b in range(e):
        m, n = map(int, input().split())
        linked[m].append(n)
        linked[n].append(m)
    
    visited = [0] * (v+1) # 리셋
    result = True
    
    for i in range(1, v+1):
        if visited[i] == 0:
            visited[i] = 1
            dfs(i)
    
    print('YES' if result else 'NO')
