import sys
input = sys.stdin.readline

T = int(input())

for _ in range(T):
    leng = int(input())
    sticker = []
    for __ in range(2):
        sticker.append(list(map(int, input().split())))
    
    jumsu = 0
    while sum(sticker[0]+sticker[1]) != 0:
        for i in range(2):
            for j in range(leng):
                if sticker[i][j] == max((map(max,sticker))) and max((map(max,sticker))) != 0:
                    jumsu += sticker[i][j]
                    for k in range(max(0, j-1), min(j+2, leng)):
                        sticker[i][k] = 0
                    if i == 0:
                        sticker[1][j] = 0
                    else:
                        sticker[0][j] = 0
    
    print(jumsu)