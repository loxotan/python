def ccw(p1, p2, p3):
    res = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
    if res > 0:
        return 1
    elif res < 0:
        return -1
    else:
        return 0

import sys
input = sys.stdin.readline

pos = []
for _ in range(3):
    x, y = map(int, input().split())
    pos.append((x,y))

print(ccw(pos[0], pos[1], pos[2]))