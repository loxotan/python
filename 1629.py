a, b, c = map(int, input().split())
a = a%c

def power(a, b, c):
    if b == 0:
        return 1
    
    elif b%2 == 0:
        half = power(a, b//2, c)
        return (half*half)%c
    
    else:
        half = power(a, (b-1)//2, c)
        return(half*half*a)%c

print(power(a,b,c))