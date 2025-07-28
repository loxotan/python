import sys
input = sys.stdin.readline

n = int(input())
gyedan = [0] + [int(input()) for _ in range(n)]

#n 번째 계단을 밟을 때
#n-1 번째 계단을 밟고 n-3번째 계단을 밟았을 때
#n-2 번째 계단을 밟았을 때

d = [0]*(n+1)
d[0] = 0
d[1] = gyedan[1]
if n>1:
    d[2] = gyedan[1]+gyedan[2]
if n>2:
    for i in range(3, n+1):
        d[i] = max(d[i-3]+gyedan[i-1], d[i-2])+gyedan[i]

print(d[n])
