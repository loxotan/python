# 1234  ++++
# 12     --   43
# 1256   ++   43
# 125     -   436
# 12578  ++   436
#       ----- 43687521

import sys
input = sys.stdin.readline

n = int(input())
num = [int(input()) for _ in range(n)]
stack = []
op = []
cnt = 1
temp = True

for i in range(n):
    while cnt <= num[i]:
        stack.append(cnt)
        op.append('+')
        cnt += 1
    if stack [-1] == num[i]:
        stack.pop()
        op.append('-')
    else:
        temp = False
        break

if temp == False:
    print('NO')
else:
    print(*op, sep='\n')