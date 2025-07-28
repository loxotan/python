import sys
input = sys.stdin.readline
import itertools

def fact(n):
    ans = 1
    for i in range(1, n+1):
        ans *= i
    return ans

T = int(input())
for _ in range(T):
    a, b = map(int, input().split())
    res = fact(b)/(fact(b-a)*fact(a))
    print(int(res))