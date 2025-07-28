N = int(input())
A = list(map(int, input().split()))

import itertools

pool = list(itertools.permutations(range(N), N))

ans = 0
for c in pool:
    temp = 0
    for i in range(1, N):
        temp += abs(A[c[i]]-A[c[i-1]])
    ans = max(ans, temp)

print(ans)