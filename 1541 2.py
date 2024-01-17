X = input().split('-')

total = 0
for i in X[0].split('+'):
    total += int(i)
for i in X[1:]:
    for j in i.split('+'):
        total -= int(j)
        
print(total)