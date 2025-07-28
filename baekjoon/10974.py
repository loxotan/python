import itertools

N = int(input())
p = itertools.permutations(list(range(1, N+1)), N)
for c in p:
    print(*c)