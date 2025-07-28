import sys
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    M, N, K = map(int, input().split())
    bat = [[0]*M for i in range(N)]
    
    #배추심기
    for _ in range(K):
        a, b = map(int, input().split())
        bat[b][a] = 1
    
    #배추가 심겨져 있으면 지렁이 올리기 (1->3)
    #주변에 지렁이가 있으면 올린척하기 (1->2)
    
    #배추밭 탐색
    for i in range(N):
        for j in range(M):
            if bat[i][j] == 1 or bat[i][j] == 2: #배추가 있으면/이미 지렁이가 올라가 있으면
                bat[i][j] = 3 if bat[i][j] == 1 else 2 #지렁이 없을때만 지렁이 올리기
                #인접한 배추 찾기
                for io in range(-1, 2): #상하
                    if 0<= i+io < N and bat[i+io][j] == 1: #배추 있으면
                        bat[i+io][j] = 2 #지렁이 올린척하기
                for jo in range(-1, 2): #좌우
                    if 0<= j+jo < M and bat[i][j+jo] == 1: #배추 있으면
                        bat[i][j+jo] = 2 #지렁이 올린척하기
            
    
    cnt = 0
    for i in range(N):#진짜 지렁이 세기
        cnt += bat[i].count(3)
        
    print(cnt)
                    