import sys
input = sys.stdin.readline

T = int(input())
T //= 2
l = []
m = 51
order = [0, 0, 2, 3, 1]
def convert(x, y):
    if x == 2 or x == 3:
        y = m - y
    a = order[x]
    return a*m + y

for _ in range(T):
    x1, y1, x2, y2 = map(int, input().split())
    a = convert(x1, y1)
    b = convert(x2, y2)
    l.append((min(a,b), max(a,b)))

ans = 0
cross = [0] * (T+1)

for i in range(T):
    for j in range(i+1, T):
        a, b = l[i]
        c, d = l[j]
        if (a < c < b < d) or (c < a < d < b):
            cross[i] += 1
            cross[j] += 1
            ans += 1

print(ans)
print(max(cross))