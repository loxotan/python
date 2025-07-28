T = int(input())

a,b,c = 0,0,0
while T>=300:
    T-= 300
    a+= 1

while T>=60:
    T-= 60
    b+= 1

while T>=10:
    T-= 10
    c+= 1

if T>0:
    print(-1)
else:
    print(a,b,c)
