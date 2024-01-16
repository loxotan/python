import sys
input = sys.stdin.readline

n = int(input())
suyeol = list(map(int, input().split()))

d= [1] * n
v= [-1] * n

for i in range(1, n):
    for j in range(i):
        if suyeol[i] > suyeol[j] and d[j]+1>d[i]:
            d[i] = d[j]+1
            v[i] = j

ans = max(d)
p = [i for i, x in enumerate(d) if x == ans][0]
print(ans)
def go(p):
    if p == -1:
        return
    go(v[p])
    print(suyeol[p], end = ' ')
go(p)
print()
