a, b = map(int, input().split())
R = int(input())

import sys
sys.setrecursionlimit(10000)

vst = [[False] * (R-i) for i in range(R)]
def flowers(x, y, t):
    if (x+1)+(y+1) < R:
        x_next, y_next = x+1, y+1
        if vst[x_next][y_next]:
            print(t+1)
            exit()
        else:
            vst[x_next][y_next] = True
            flowers(x_next, y_next, t+1)
    else:
        x_next, y_next = x//2, y//2
        if vst[x_next][y_next]:
            print(t+1)
            exit()
        else:
            vst[x_next][y_next] = True
            flowers(x_next, y_next, t+1)

vst[a][b] = True
flowers(a, b, 0)