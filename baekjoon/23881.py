import sys
input = sys.stdin.readline

n, k = map(int, input().split())
nums = list(map(int, input().split()))

cnt = 0
for i in range(n-1, 0, -1):
    x = max(nums[:i+1])
    y = nums[i]
    if x != y:
        nums[i], nums[nums.index(x)] = x, y
        cnt += 1
    if cnt == k:
        print(y, x)
        break
else:
    print(-1)