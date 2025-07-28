n, m = map(int, input().split())
S = list(map(int, input().split()))
S.sort()
visited = [False] * n
import sys
sys.setrecursionlimit(10000)
import copy

def dfs(A, idx):
    if idx == m:
        ans.add(tuple(A[1:]))
        return
        
    for i in range(n):
        if A[-1]<=S[i] and visited[i] == False:
            A.append(S[i])
            visited[i] = True
            dfs(A, idx+1)
            A.pop()
            visited[i] = False
        
A = [0]
ans = set()
dfs(A, 0)
for i in sorted(ans):
    print(*i)

    