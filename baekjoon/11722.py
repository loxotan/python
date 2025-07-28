import sys
input = sys.stdin.readline

n = int(input())
suyeol = list(map(int, input().split()))

d = [1] * n

for i in range(1, n):
    for j in range(i):
        if suyeol[i] < suyeol[j]:
            d[i] = max(d[i], d[j]+1)

print(max(d))