n = int(input())

#소인수분해

def how_many_soinsu(n):
    k = 2
    soinsu = []
    while n>k:
        if n%k == 0:
            soinsu.append(k)
            n //= k
        else:
            k+=1
    return len(soinsu)    

#소인수가 홀수인 경우 1->0, 짝수인 경우 1->0->1
arr = [1] * n
for i in range(2, n+1):
    if how_many_soinsu(i)%2==1:
        arr[i-1] = 0

print(sum(arr))

# 1 1 1 1 1
# 1 0 1 0 1
# 1 0 0 0 1
# 1 0 0 1 1
# 1 0 0 1 0
