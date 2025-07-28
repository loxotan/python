n, m = map(int, input().split())
S = list(map(int, input().split()))
S.sort()

import sys
sys.setrecursionlimit(10000)

def dfs(A, idx):
    if idx == m:
        ans.add(tuple(A[1:]))
        return
        
    for i in range(n):
        if A[-1] <= S[i]:
            A.append(S[i])
            dfs(A, idx+1)
            A.pop()

        
A = [0]
ans = set()
dfs(A, 0)
for i in sorted(ans):
    print(*i)

    