import sys
input = sys.stdin.readline

n, m = map(int, input().split())

out = [0]*(n+1)

for _ in range(m):
    r, s, e = map(int, input().split())
    if s >= out[r]:
        print('YES')
        out[r] = e
    else:
        print('NO')