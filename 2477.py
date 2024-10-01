import sys
input = sys.stdin.readline

def calculate_area(pos):
    s1, s2 = 0, 0
    for i in range(len(pos)-1):
        s1 += pos[i][0] * pos[i+1][1]
        s2 += pos[i+1][0] * pos[i][1]
    return abs(s1-s2) /2

density = int(input())

pos = [(0,0)]
for _ in range(6):
    dir, meter = map(int, input().split())
    x, y = pos[-1][0], pos[-1][1]
    if dir == 1:
        pos.append((x+meter, y))
    elif dir == 2:
        pos.append((x-meter, y))
    elif dir == 3:
        pos.append((x, y-meter))
    else:
        pos.append((x, y+meter))

area = int(calculate_area(pos))
print(density * area)