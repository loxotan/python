"""def w(a, b, c):
    if a<= 0 or b<=0 or c<=0:
        return 1
    elif a>20 or b>20 or c>20:
        return (w(20, 20, 20))
    elif a<b and b<c:
        return w(a,b,c-1) + w(a,b-1,c-1) - w(a,b-1,c)
    else:
        return w(a-1,b,c)+w(a-1,b-1,c)+w(a-1,b,c-1)-w(a-1,b-1,c-1)"""

import sys
input = sys.stdin.readline

while True:
    a, b, c = map(int, input().split())
    if a == -1 and b == -1 and c == -1:
        break
    
    dp = [[[1 for _ in range(21)] for __ in range(21)] for ___ in range(21)]
    for A in range(1, min(a+1, 21)):
        for B in range(1, min(b+1, 21)):
            for C in range(1, min(c+1, 21)):
                if A<B and B<C:
                    dp[A][B][C] = dp[A][B][C-1] + dp[A][B-1][C-1] - dp[A][B-1][C]
                else:
                    dp[A][B][C] = dp[A-1][B][C] + dp[A-1][B-1][C] + dp[A-1][B][C-1] - dp[A-1][B-1][C-1]
    
    if a>20 or b>20 or c>20:
        print(f'w({a}, {b}, {c}) = {dp[20][20][20]}')
    else:
        print(f'w({a}, {b}, {c}) = {dp[a][b][c]}')
