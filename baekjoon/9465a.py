import sys
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    N = int(input())
    R1 = list(map(int, input().split()))
    R2 = list(map(int, input().split()))
    
    D = [[0,0,0] for _ in range(N)]
    D[0][1] = R1[0]
    D[0][2] = R2[0]
    
    for i in range(1, N):
        D[i][0] = max(D[i-1])
        D[i][1] = max(R1[i] + D[i-1][0], R1[i] + D[i-1][2])
        D[i][2] = max(R2[i] + D[i-1][0], R2[i] + D[i-1][1])
    
    print(max(D[N-1]))