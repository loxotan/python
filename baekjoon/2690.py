a, b = map(int, input().split())

def gcd(n, m):
    if m > n:
        n,m = m,n
    if n%m == 0:
        return m
    else:
        return gcd(n%m, m)
    
print(gcd(a,b))
print(a*b//gcd(a, b))