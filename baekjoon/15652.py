n, m = map(int, input().split())
import sys
sys.setrecursionlimit(10000)

def dfs(A, idx):
    if idx == m:
        print(*A[1:])
        return
        
    i = A[-1] 
    for x in range(i, n+1):
        A.append(x)
        dfs(A, idx+1)
        A.pop()
        
        
A = [1]        
dfs(A, 0)
    