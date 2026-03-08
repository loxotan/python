import sys 
input = sys.stdin.readline

T = int(input())
tanfuru = list(map(int, input().split()))

cnt = [0] * (10)
kind = 0
L = 0
ans = 0

for R in range(T):
    if cnt[tanfuru[R]] == 0:
        kind += 1
    cnt[tanfuru[R]] += 1
    
    while kind > 2:
        cnt[tanfuru[L]] -= 1
        if cnt[tanfuru[L]] == 0:
            kind -= 1
        L += 1

    ans = max(ans, R-L+1)
print(ans)