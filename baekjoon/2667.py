import sys
input = sys.stdin.readline

n = int(input())
danji = [[] for _ in range(n)]
for i in range(n):
    line = str(input())
    for j in range(n):
        danji[i].append(int(line[j]))

def dfs(i, j):
    global vst, cnt
    vst[i][j] = True
    cnt += 1
    dir = [(1,0), (-1,0), (0,1), (0,-1)]
    for io, jo in dir:
        if 0<=i+io<n and 0<=j+jo<n and not vst[i+io][j+jo] and danji[i+io][j+jo] == 1:
            dfs(i+io, j+jo)
 
vst = [[False]*n for _ in range(n)]     
danjisu = []  
for i in range(n):
    for j in range(n):
        if not vst[i][j] and danji[i][j] == 1:
            cnt = 0
            dfs(i,j)
            danjisu.append(cnt)
            
danjisu.sort()
print(len(danjisu))
print(*danjisu, sep = '\n')

