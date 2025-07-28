daneo = str(input())

while len(daneo)>0:
    if len(daneo)>10:
        print(daneo[:10])
        daneo = daneo[10:]
    else:
        print(daneo)
        break