nums = [False, False] + [True] * 1100001
pal_primes = []
for i in range(1100002):
    if nums[i]:
        if str(i) == str(i)[::-1]:
            pal_primes.append(i)
        for j in range(i+i, 1100002, i):
            nums[j] = False

print(pal_primes)
n = int(input())
for pal in pal_primes:
    if pal > n:
        print(pal)
        break