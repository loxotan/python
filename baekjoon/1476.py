e, s, m = map(int, input().split())
n=1
while n>0:
    if (n-e)%15 == 0 and (n-s)%28 == 0 and (n-m)%19 == 0:
        print(n)
        break
    else:
        n += 1
        
