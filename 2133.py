n = int(input())

d = [0]*(n+1)
d[1] = 0
if n>1:
    d[2] = 3
for i in range(4, n+1, 2):
    d[i] = d[i-2]*3
    for j in range(4, i, 2):
        d[i] += d[i-j]*2
    d[i]+=2

print(d[n])