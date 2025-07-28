S = input().rstrip()
res = []
tag = 0
temp = []

for word in S:
    if word == '<':
        if len(temp) != 0:
            res.append(''.join(temp[::-1]))
            temp.clear()
        temp.append('<')
        tag = 1
    elif word == '>':
        temp.append('>')
        res.append(''.join(temp))
        temp.clear()
        tag = 0
    elif tag == 0 and word == ' ':
        res.append(''.join(temp[::-1]))
        res.append(' ')
        temp.clear()
    else:
        temp.append(word)
if len(temp) != 0:
    res.append(''.join(temp[::-1]))

print(*res, sep = '')
    