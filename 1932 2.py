import sys
input = sys.stdin.readline

n = int(input())
tri = [list(map(int, input().split())) for _ in range(n)]

for i in range(1, n):
    tri[i][0] = tri[i-1][0] + tri[i][0]
    tri[i][i] = tri[i-1][i-1] + tri[i][i]
    for j in range(1, i):
        tri[i][j] = max(tri[i-1][j-1], tri[i-1][j]) + tri[i][j]

print(max(tri[n-1]))