m, n = map(int, input().split())

#소수 판별
def is_sosu(n):
    if n == 1:
        return False
    for s in range(2, int(n**0.5+1)):
        if n%s == 0:
            return False
    else:
        return True
    
for n in range(m, n+1):
    if is_sosu(n) == True:
        print(n)
