import sys
input = sys.stdin.readline

n = int(input())

for _ in range(n):
    line = str(input())
    cnt = 0
    scr = 0
    for i in range(len(line)):
        if line[i] == 'O':
            cnt += 1
            scr += cnt
        else:
            cnt = 0
    print(scr)