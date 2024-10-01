import sys
input = sys.stdin.readline

def inside(x1, y1, x2, y2, r):
    dist = ((x1-x2)**2+(y1-y2)**2)
    return r**2 > dist

def xor(a, b):
    return a != b

T = int(input())
for _ in range(T):
    x1,y1,x2,y2 = map(int, input().split())
    n = int(input())
    ans = 0
    for _ in range(n):
        x, y, r = map(int, input().split())
        start = inside(x1, y1, x, y, r)
        end = inside(x2, y2, x, y, r)
        if xor(start, end):
            ans += 1
    print(ans)