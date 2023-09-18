import sys
input = sys.stdin.readline

T = int(input())

def geoju(a, b):
    if a==0:
        return b
    else:
        return sum([geoju(a-1,i) for i in range(b+1)])

for _ in range(T):
    k = int(input())
    n = int(input())
    print(geoju(k, n))