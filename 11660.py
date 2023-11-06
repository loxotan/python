import sys
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [[0]*(n+1)] + [[0] + list(map(int, input().split())) for _ in range(n)]

for i in range(n+1):
    for j in range(1, n+1):
        graph[i][j] += graph[i][j-1]
        
for _ in range(m):
    y1, x1, y2, x2 = map(int, input().split())
    temp = 0
    for i in range(y1, y2+1):
        temp += graph[i][x2]-graph[i][x1-1]
    print(temp)