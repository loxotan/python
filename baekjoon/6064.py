import sys
input = sys.stdin.readline

def calculate(m, n, x, y):
    i = x
    while i<=m*n:
        if (i-x)%m == 0 and (i-y)%n == 0: 
            return i
        i += m
    return -1

T = int(input())
for _ in range(T):
    m, n, x, y = map(int, input().split())
    print(calculate(m, n, x, y))
            