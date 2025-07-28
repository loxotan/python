import sys
input = sys.stdin.readline
from collections import deque

T = int(input())

for _ in range(T):
    cnt = 0
    go = 0
    cmd = input().rstrip()
    n = int(input())
    lst = deque(input().strip()[1:-1].split(','))
    
    for j in cmd:
        if j == 'R' and lst:
            cnt += 1
        elif j == 'D':
            if lst:
                if lst[0] == '':
                    print('error')
                    go = 1
                    break
                if cnt%2 == 1:
                    lst.pop()
                else:
                    lst.popleft()
            else:
                print('error')
                go = 1
                break
        
    
    if go == 0:
        if cnt%2 == 1:
            lst.reverse()
        print('['+','.join(lst)+']')
        