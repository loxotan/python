import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)

T = int(input())

def dfs(arr, row, col, N, M):
    if arr[row][col] == 1:
        arr[row][col] = 0
        for di, dj in [[0,1], [0,-1], [1,0], [-1,0]]:
            ni = row + di
            nj = col + dj
            if 0 <= ni < N and 0 <= nj < M:
               dfs(arr, ni, nj, N, M)
        return 1
    return 0
    
for _ in range(T):
    M, N, K = map(int, input().split())
    bat = [[0]*M for i in range(N)]
    
    #배추심기
    for _ in range(K):
        a, b = map(int, input().split())
        bat[b][a] = 1
    
    cnt = 0
    for i in range(N):
        for j in range(M):
            cnt += dfs(bat, i, j, N, M)
    
    print(cnt)