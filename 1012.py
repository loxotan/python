import sys
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    M, N, K = map(int, input().split())
    bat = [[0]*M for i in range(N)]
    
    #배추심기
    for _ in range(K):
        a, b = map(int, input().split())
        bat[a][b] = 1
    
    #배추가 심겨져 있으면 지렁이 올리기 (1->3)
    #주변에 지렁이가 있으면 올린척하기 (1->2)
    
    #배추밭 탐색
    for i in range(N):
        for j in range(M):
            