import sys
input = sys.stdin.readline

n = int(input())
stack = []
for i in range(n):
    cmd = input().split()
    
    #push
    if cmd[0] == 'push':
        stack.append(cmd[1])
    
    #pop
    if cmd[0] == 'pop':
        print(-1 if len(stack)== 0 else stack.pop())
    
    #size
    if cmd[0] == 'size':
        print(len(stack))
    
    #empty
    if cmd[0] == 'empty':
        print(1 if len(stack)==0 else 0)
    
    #top
    if cmd[0] == 'top':
        print(-1 if len(stack)==0 else stack[-1])