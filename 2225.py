n, k = map(int, input().split())
# n+1 H k-1 = n+k-1 C k-1

def fact(n):
    d=1
    for i in range(2, n+1):
        d *= i
    return d

result = fact(n+k-1)//(fact(n)*fact(k-1))
print(result%1000000000)