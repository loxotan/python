import sys
input = sys.stdin.readline

n = int(input())
suyeol = [0] + list(map(int, input().split()))

d = [[] for _ in range(n+1)]
d[1].append(suyeol[1])
if n>1:
    for i in range(2, n+1):
        for j in range(1, i):
            if suyeol[i] > d[j][-1]:
                d[j].append(suyeol[i])
        d[i].append(suyeol[i])
        
d_len = list(map(len, d))
print(max(d_len))
