import sys
input = sys.stdin.readline

K = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

a.sort(reverse=True)
b.sort()

cnt = 0
for i in range(K):
    cnt += a[i] * b[i]
    
print(cnt)