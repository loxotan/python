import sys
input = sys.stdin.readline

T = int(input())
for t in range(T):
    S = input().rstrip()
    S += ' '
    stack = []
    for word in S:
        if word == ' ':
            while stack:
                print(stack.pop(), end = '')
            print(' ', end = '')
        else:
            stack.append(word)
            