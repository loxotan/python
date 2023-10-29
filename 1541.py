X = input().rstrip() + '+'

plus = []
minus = []
temp = ''
minus_cnt = 0
for x in X: # - 나왔을 때 다음 - 까지 괄호하면 최소
    if minus_cnt == 1 and (x == '-' or x == '+'):
        if x == '-':
            minus_cnt = 0
        minus.append(int(temp))
        temp = ''
    elif minus_cnt == 0 and (x == '-' or x == '+'):
        if x == '-':
            minus_cnt = 1
        plus.append(int(temp))
        temp = ''
    else:
        temp += x
        
print(sum(plus) - sum(minus))