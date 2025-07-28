n, m = map(int, input().split())
A = list(map(int, input().split()))
A.sort()

import itertools
C = list(itertools.combinations(A, m))
for i in range(len(C)):
    print(*C[i])