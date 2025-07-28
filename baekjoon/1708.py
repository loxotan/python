import sys
input = sys.stdin.readline

n = int(input())
pos = []
for _ in range(n):
    x, y = map(int, input().split())
    pos.append((x,y))
pos = sorted(pos, key=lambda p:(p[0], p[1]))

def ccw(p1, p2, p3):
    res = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
    if res > 0:
        return True
    else:
        return False

def hull(pos, reverse = False):
    if reverse == True:
        pos = pos [::-1]
    
    chain = []
    for p in pos:
        while len(chain)>=2 and not ccw(chain[-2], chain[-1], p):
            chain.pop()
        chain.append(p)
    
    return chain

upper = hull(pos)
lower = hull(pos, reverse = True)
whole = upper + lower[1:-1]
print(len(whole))