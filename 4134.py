import sys

n = int(input())
#소수 판별
def is_sosu(n):
    for s in range(2, int(n**0.5+1)):
        if n%s == 0:
            return False
    else:
        return True

for i in range(n):
    num = int(sys.stdin.readline())
    if num == 0 or num == 1:
        print (2)
    else:
        while True:
            if is_sosu(num) == False:
                num+=1
            else:
                print(num)
                break
