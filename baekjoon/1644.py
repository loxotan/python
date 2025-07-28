import sys
sys.setrecursionlimit(10000)

n = int(input())

# 4_000_000 까지 모든 소수
nums = [False, False] + [True]*n
primes = []
lst = [0]
for i in range(2, n+2):
    if nums[i]:
        primes.append(i)
        lst.append(lst[-1] + i)
        for j in range(i*i, n+2, i):
            nums[j] = False
                
l, r, cnt = 0, 1, 0

while r < len(lst):
    psum = lst[r] - lst[l]
    if psum < n:
        r += 1
    elif psum > n:
        l += 1
    else:
        cnt += 1
        l += 1

print(cnt)