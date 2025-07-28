n,x = map(int, input().split())
nums = list(map(int, input().split()))

if sum(nums) % x == 0:
    print(1)
else:
    print(0)