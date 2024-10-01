import sys
input = sys.stdin.readline

INF = int(1e9)

n = int(input())
pos = []
for _ in range(n):
    x, y = map(int, input().split())
    pos.append((x,y))

pos = sorted(pos, key=lambda p:(p[0], p[1]))

def distance(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def shortest_line(line, d):
    min_dist = d
    for i in range(len(line)):
        for j in range(i+1, len(line)):
            if (line[j][1] - line[i][1])**2 >= min_dist:
                break
            min_dist = min(min_dist, distance(line[i], line[j]))
    return min_dist


def search(pos, n): 
    if n<=3:
        dist = INF
        for i in range(n):
            for j in range(i+1, n):
                dist = min(dist, distance(pos[i], pos[j]))
        return dist
    
    mid = n//2
    x_mid = pos[mid][0]
    
    d_left = search(pos[:mid], mid)
    d_right = search(pos[mid:], n-mid)
    d = min(d_left, d_right)
    
    line = []
    for p in pos:
        if abs(p[0] - x_mid)**2 < d:
            line.append(p)
    line = sorted(line, key = lambda p:p[1])
    
    return min(d, shortest_line(line, d))


def main():
    print(int(search(pos, n)))

main()
