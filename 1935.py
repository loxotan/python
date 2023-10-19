import sys
input = sys.stdin.readline

N = int(input())
sik = input().rstrip()
stack = []
dic = {}
for i in range(N):
    dic.update({chr(i+65):int(input())})
    
for x in sik:
    if x == '+' or x == '-' or x == '*' or x == '/':
        a = stack.pop()
        b = stack.pop()
        if x == '+':
            stack.append(b+a)
        elif x == '-':
            stack.append(b-a)
        elif x == '*':
            stack.append(b*a)
        elif x == '/':
            stack.append(b/a)
    else:
        stack.append(dic[x])
         
print("{:.2f}".format(*stack))