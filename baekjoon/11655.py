n = list(input())
new_n = []
for i in range(len(n)):
    if n[i] == ' ':
        new_n.append(' ')
    else:
        if 57>=ord(n[i])>=48:
            new_n.append(n[i])
        elif 109>=ord(n[i])>=97 or 77>=ord(n[i])>=65:
            new_n.append(chr(ord(n[i])+13))
        else:
            new_n.append(chr(ord(n[i])-13))
        
print(*new_n, sep='')