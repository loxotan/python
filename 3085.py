<<<<<<< HEAD
import sys
input = sys.stdin.readline

n = int(input())
arr = [list(input()) for _ in range(n)]]
=======
from sys import stdin
input = stdin.readline

N = int(input())
arr = list([] for _ in range(N))
for i in range(N):
    x = input().rstrip()
    for j in range(N):
        arr[i].append(x[j])
        
def check():
    mx_x = 0
    mx_y = 0
    for i in range(N):
        cnt_x = 1
        cnt_y = 1
        for j in range(1, N):
            if arr[i][j-1] == arr[i][j]: #행에서 제일 긴 사탕 수
                cnt_x += 1
            else:
                mx_x = max(cnt_x, mx_x)
                cnt_x = 1
            if arr[j-1][i] == arr[j][i]: #열에서 제일 긴 사탕 수
                cnt_y += 1
            else:
                mx_y = max(cnt_y, mx_y)
                cnt_y = 1
        mx_x, mx_y = max(cnt_x, mx_x), max(cnt_y, mx_y)
    mx = max(mx_x, mx_y)               
    return mx

mx_xo = 0
mx_yo = 0
for i in range(N):
    for j in range(1, N):
        if arr[i][j-1] != arr[i][j]: #행에서 사탕 바꾸기
            arr[i][j-1], arr[i][j] = arr[i][j], arr[i][j-1]
            mx_xo = max(mx_xo, check())
            arr[i][j-1], arr[i][j] = arr[i][j], arr[i][j-1]
        if arr[j-1][i] != arr[j][i]:
            arr[j-1][i], arr[j][i] = arr[j][i], arr[j-1][i]
            mx_yo = max(mx_yo, check())
            arr[j-1][i], arr[j][i] = arr[j][i], arr[j-1][i]
mx = max(mx_yo, mx_xo)

print(mx)
>>>>>>> 8c40a3bddbe0b4d1a1a7631360befe6b23537378
