import sys
input = sys.stdin.readline

n, m = map(int, input().split())
friends = []
for _ in range(m):
    a, b = map(int, input().split())
    if a > b:
        a, b = b, a
    if [a,b] not in friends:
        friends.append([a,b])

