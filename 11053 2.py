import sys
input = sys.stdin.readline

n = int(input())
suyeol = list(map(int, input().split()))
reverse = suyeol[::-1]

d = [1] * n
c = [1] * n

for i in range(1, n):
    for j in range(i):
        if suyeol[i] > suyeol[j]:
            d[i] = max(d[i], d[j]+1)
        if reverse[i] > reverse[j]:
            c[i] = max(c[i], c[j]+1)

result = [0 for i in range(n)]
for i in range(n):
    result[i] = d[i] + c[n-i-1]-1

print(max(result))