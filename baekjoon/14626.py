ISBN = str(input())

s = 0
for i in range(13):
    if ISBN[i] == '*':
        n = i
    else:
        if i % 2 == 0:
            s += int(ISBN[i])
            s %= 10
        else:
            s += 3 * int(ISBN[i])
            s %= 10

if n % 2 == 0:
    x = (10 - s)%10
else:
    p = 10-s
    if p == 1:
        x = 7
    elif p == 2:
        x = 4
    elif p == 4:
        x = 8
    elif p == 5:
        x = 5
    elif p == 7:
        x = 9
    elif p == 8:
        x = 6
    elif p == 10 or p == 0:
        x = 0
    else:
        x = p // 3

print(x)
