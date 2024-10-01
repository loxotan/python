import sys
input = sys.stdin.readline

def ccw(p1, p2, p3):
    res = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
    if res > 0:
        return True
    else:
        return False

def chaining(pos, reverse = False):
    chain = []
    if reverse:
        pos = pos[::-1]
    
    for p in pos:
        while len(chain)>=2 and not ccw(chain[-2], chain[-1], p):
            chain.pop()
        chain.append(p)
    
    return chain

def calculate_area(pos):
    s1, s2 = 0, 0
    for i in range(len(pos)-1):
        s1 += pos[i][0] * pos[i+1][1]
        s2 += pos[i+1][0] * pos[i][1]
    return abs(s1-s2) /2

def main():
    n = int(input())

    pos = []
    for _ in range(n):
        x, y = map(int, input().split())
        pos.append((x, y))
    pos = sorted(pos, key=lambda p:(p[0], p[1]))
    
    if len(pos) < 3:
        print(0)
        return

    upper = chaining(pos)
    lower = chaining(pos, reverse = True)
    chain = upper + lower[1:-1]
    chain.append(chain[0])

    print(int(calculate_area(chain)//50))

main()
