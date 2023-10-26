n, m = map(int, input().split())
import itertools
A = list(itertools.product(range(1, n+1), repeat = m))
for i in range(len(A)):
    print(*A[i])
