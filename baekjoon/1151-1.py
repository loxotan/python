import sys
input = sys.stdin.readline

grim = [list(input().split()) for _ in range(15)]

for i in range(15):
    for j in range(15):
        if grim[i][j] == 'w':
            print('chunbae')
            exit()
        elif grim[i][j] == 'b':
            print('nabi')
            exit()
        elif grim[i][j] == 'g':
            print('yeongcheol')
            exit()
        else:
            continue
        