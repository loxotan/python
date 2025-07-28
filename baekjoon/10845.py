import sys
input = sys.stdin.readline

from collections import deque


n = int(input())
stack = deque([])

for i in range(n):
    cmd = list(input().split())
    
    #push
    if cmd[0] == 'push':
        stack.append(cmd[1])
    
    #pop
    elif cmd[0] == 'pop':
        print(-1 if len(stack) == 0 else stack.popleft())
    
    #size
    elif cmd[0] == 'size':
        print(len(stack))
    
    #empty
    elif cmd[0] == 'empty':
        print(1 if len(stack) == 0 else 0)
    
    #front
    elif cmd[0] =='front':
        print(-1 if len(stack) == 0 else stack[0])
    
    #back
    elif cmd[0] == 'back':
        print(-1 if len(stack) == 0 else stack[-1])