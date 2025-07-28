import sys
input = sys.stdin.readline

n = int(input())
for _ in range(n):
    v, e = map(int, input().split())
    print(2-v+e)