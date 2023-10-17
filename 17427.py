import sys
input = sys.stdin.readline

n = int(input())

def g(n):
    return sum(k*(n//k) for k in range(1, n+1))

for _ in range(n):
    x = int(input())
    print(g(x))