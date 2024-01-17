import sys
input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))
dp = [[0]*N for _ in range(N)]

for i in range(N):
    dp[i][i] = 1
    if i+1 < N and A[i] == A[i+1]:
        dp[i][i+1] = 1
        
for i in range(N-1, -1, -1):
    for j in range(i+1, N):
        if A[i] == A[j] and dp[i+1][j-1] == 1:
            dp[i][j] = 1
            
M = int(input())
for T in range(M):
    s, e = map(int, input().split())
    print(dp[s-1][e-1])