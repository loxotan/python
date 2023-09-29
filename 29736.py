a,b = map(int, input().split())
k,x = map(int, input().split())

if a>k+x or b<k-x:
    print("IMPOSSIBLE")
elif b>k+x:
    if a<k-x:
        print(2*x+1)
    else:
        print(k+x-a+1)
else:
    if a<k-x:
        print(b-(k-x)+1)
    else:
        print(b-a+1)