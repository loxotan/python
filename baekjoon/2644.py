import sys
input = sys.stdin.readline

n = int(input())
q = list(map(int, input().split()))
m = int(input())
graph = [0] + [[] for _ in range(n)]
for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

vst = [False] * (n+1)
broken = 0
def dfs(start, end, depth):
    global broken
    if start == end:
        print(depth)
        broken = 1
        return 
    
    for x in graph[start]:
        if not vst[x]:
            vst[x] = True
            dfs(x, end, depth+1)
            vst[x] = False
            
dfs(q[0], q[1], 0)
if broken == 0:
    print(-1)