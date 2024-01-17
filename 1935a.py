import sys
input = sys.stdin.readline

N = int(input())
S = input().rstrip()
word = []
stack = []

for n in range(N):
    word.append(int(input()))

for letter in S:
    if letter in ['+', '-', '*', '/']:
        a = stack.pop()
        b = stack.pop()
        if letter == '+':
            stack.append(b+a)
        elif letter == '-':
            stack.append(b-a)
        elif letter == '*':
            stack.append(b*a)
        elif letter == '/':
            stack.append(float("{:.2f}".format(b/a)))
    else:
        stack.append(word[ord(letter)-65])
        
print("{:.2f}".format(*stack))