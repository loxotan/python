N = int(input())
line = input().rstrip()

lower = 'roygbiv'
upper = 'ROYGBIV'

L = set()
U = set()

for x in line:
    if x in lower:
        L.add(x)
    elif x in upper:
        U.add(x)

if len(L) == 7 and len(U) == 7:
    print('YeS')
elif len(L) == 7:
    print('yes')
elif len(U) == 7:
    print('YES')
else:
    print('NO!')
        