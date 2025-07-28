import sys
input = sys.stdin.readline

n, k = map(int, input().split())
bag = [0 for i in range(k+1)]
things = []

for _ in range(n):
    w, v = map(int, input().split())
    things.append((w,v))

for w, v in things:
    for i in range(k, w-1, -1):
        bag[i] = max(bag[i], bag[i-w]+v)

print(max(bag))