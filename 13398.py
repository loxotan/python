import sys
input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))

D1 = [1] * N
D2 = [1] * N

# i번째에서 끝나는 연속합
D1[0] = A[0]
for i in range(1, N):
    D1[i] = max(A[i], D1[i-1] + A[i])
    
#i번째에서 시작하는 연속합
D2[-1] = A[-1]
for i in range(N-2, -1, -1):
    D2[i] = max(A[i], D2[i+1] + A[i])
    
#안 뺀 연속합의 최대
ans = max(D1)

#하나 뺀 연속합의 최대
for i in range(1,N-1):
    if ans < D1[i-1]+D2[i+1]:
        ans = D1[i-1]+D2[i+1]
        
print(ans)