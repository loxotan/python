import sys
input = sys.stdin.readline

T = int(input())

geoju = [[0]*15 for _ in range(15)]
for i in range(15):
    geoju[0][i] = i
for i in range(1, 15):
    for j in range(1, 15):
        geoju[i][j] = sum((geoju[i-1])[:j+1])


for _ in range(T):
    k = int(input())
    n = int(input())
    print(geoju[k][n])