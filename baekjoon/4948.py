n = 1

def is_sosu(n):
    if n == 1:
        return False
    for s in range(2, int(n**0.5+1)):
        if n%s == 0:
            return False
    else:
        return True

while n>0:
    n = int(input())
    if n == 0:
        break
    result = 0
    for i in range(n+1, 2*n+1):
        if is_sosu(i) == True:
            result += 1
    print(result)