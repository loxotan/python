import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
friends = [[False]*(n+1) for _ in range(n+1)]
for _ in range(m):
    a, b = map(int, input().split())
    friends[a][b] = True
    friends[b][a] = True

def kevin(a, b):#a 부터 b까지 찾는데 걸린 수 return하기
    q = deque()
    q.append((a, 0))
    visited = [False] * (n+1)
    visited[a] = True
    while q:
        v, d = q.popleft()
        if v == b:
            return d
        for i in range(1, n+1):
            if not visited[i] and friends[v][i]:
                visited[i] = True
                q.append((i, d+1))
        
kevin_list = [0]*(n+1)
for i in range(1, n+1):
    for j in range(1, n+1):
        kevin_list[i] += kevin(i, j)

kevin_list.pop(0)
print(kevin_list.index(min(kevin_list))+1)