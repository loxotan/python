n, m = map(int, input().split())
S = list(map(int, input().split()))
S.sort()

import sys
sys.setrecursionlimit(10000)

def dfs(A, idx):
    if idx == m:
        print(*A[1:])
        return
        
    i = A[-1] 
    for x in S:
        if x >= i:
            A.append(x)
            dfs(A, idx+1)
            A.pop()
        
A = [1]        
dfs(A, 0)
    