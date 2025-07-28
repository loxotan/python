import sys
input = sys.stdin.readline

n = int(input())

pos = []
for _ in range(n):
    x, y = map(int, input().split())
    pos.append((x, y))
pos.append(pos[0])

#신발끈
s1, s2 = 0, 0
for i in range(n):
    s1 += pos[i][0] * pos[i+1][1]
    s2 += pos[i+1][0] * pos[i][1]

print((1/2)*abs(s1-s2))