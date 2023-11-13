import sys
input = sys.stdin.readline

stack = []
T = int(input())
for _ in range(T):
    a, b = map(int, input().split())
    if a == 1:
        stack.append(b)
    if a == 2 and stack:
        k = max(stack)
        p = k-b
        if p > 0:
            check = False
            for i in range(len(stack)):
                if check:
                    stack[i] = p
                if not check and stack[i] > p:
                    stack[i] = p
                    check = True
                

print(sum(stack))
