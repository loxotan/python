infix = str(input())

op = ['+', '-', '*', '/']

def precedence(op):
    if op == '(' or op == ')':
        return 0
    elif op == '+' or op == '-':
        return 1
    elif op =='*' or op == '/':
        return 2

ops = []
for i in range(len(infix)):
    x = infix[i]
    if x not in op + ['(', ')']:
        print(x, end = '')
    elif x in op:
        while ops:
            if precedence(x) > precedence(ops[-1]):
                break
            else:
                print(ops.pop(), end='')
        ops.append(x)
    elif x == '(':
        ops.append(x)
    else:
        while ops and ops[-1] != '(':
            print(ops.pop(), end='')
        ops.pop()

while ops:
    v = ops.pop()
    print(v, end = '')
