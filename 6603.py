
import sys
input = sys.stdin.readline

def dfs(depth, idx):
    if depth == 6:
        print(*out)
        return
    for i in range(idx, k):
        out.append(s[i])
        dfs(depth+1, i+1)
        out.pop()
    
while True:
    n = list(map(int, input().split()))
   
    k = n[0]
    s = n[1:]
    out = []
    dfs(0, 0)
    if n[0] == 0:
        exit()
    print()
