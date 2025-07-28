import sys
input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))
d = [0] * N
d[0] = A[0]

for i in range(1, N):
    d[i] = max(A[i], d[i-1] + A[i])
    
print(max(d))
