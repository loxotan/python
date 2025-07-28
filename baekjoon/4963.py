import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)

dir = []
for i in range(-1, 2):
    for j in range(-1, 2):
        dir.append((i,j))
        
def dfs(i, j):
    global vst, dir
    vst[i][j] = True
    for io, jo in dir:
        if 0<=i+io<h and 0<=j+jo<w and not vst[i+io][j+jo] and jido[i+io][j+jo] == 1:
            dfs(i+io, j+jo)

while True:
    w, h = map(int, input().split())
    if w == 0:
        break
    
    jido = []
    for _ in range(h):
        jido.append(list(map(int, input().split())))
    
    vst = [[False] * (w) for _ in range(h)]
    cnt = 0
    for m in range(h):
        for n in range(w):
            if not vst[m][n] and jido[m][n] == 1:
                dfs(m, n)
                cnt += 1
                
    print(cnt)
                 
