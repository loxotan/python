n, m = map(int, input().split())
S = list(map(int, input().split()))
S.sort()
visited = [False] * n
import sys
sys.setrecursionlimit(10000)

def dfs(A, idx):
    if idx == m:
        ans.add(tuple(A))
        return
        
    for i in range(n):
        A.append(S[i])
        dfs(A, idx+1)
        A.pop()

        
A = []
ans = set()
dfs(A, 0)
for i in sorted(ans):
    print(*i)

    