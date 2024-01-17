import math
import sys
import itertools
input = sys.stdin.readline

N = int(input())
for _ in range(N):
    S1 = list(map(int, input().split()))
    S = S1[1:]
    case = list(itertools.combinations(S, 2))
    gcds = []
    for (a, b) in case:
        g = math.gcd(a, b)
        gcds.append(g)
    print(sum(gcds))