import sys

n = int(input())
stack = []

for i in range(n):
    com = sys.stdin.readline().rstrip()
    
    if len(com)>2:
        stack.append(int(com[2:]))
    elif com == '2':
        if len(stack) == 0:
            print(-1)
        else:
            print(stack.pop())
    elif com == '3':
        print(len(stack))
    elif com == '4':
        print(1 if len(stack)==0 else 0)
    elif com == '5':
        if len(stack)==0:
            print(-1)
        else:
            print(stack[-1])
            

