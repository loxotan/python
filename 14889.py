import sys
from itertools import combinations

input = sys.stdin.readline
N = int(input())
graph = [list(map(int, input().split())) for _ in range(N)]

pool = list(range(N))
c = []
for i in combinations(pool, N//2):
    c.append(i)


ans = 1e9
for case in c:
    if 0 in case:
        a, b = 0, 0
        for v1, v2 in combinations(case, 2):
            a += graph[v1][v2] + graph[v2][v1]
        for w1, w2 in combinations(list(set(pool)-set(case)), 2):
            b += graph[w1][w2] + graph[w2][w1]
        ans = min(ans, abs(a-b))

print(ans)
