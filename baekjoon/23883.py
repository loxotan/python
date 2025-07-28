n, k = map(int, input().split())
nums = list(map(int, input().split()))
nums2 = sorted(nums)
d = {}

for i, l in enumerate(nums):
    d[l] = i
    
cnt = 0
for i in range(n-1, -1, -1):
    if nums2[i] != nums[i]:
        temp=[nums[i], nums2[i]]
        nums[i], nums[d[nums2[i]]] = nums[d[nums2[i]]], nums[i]
        d[temp[0]], d[temp[1]] = d[temp[1]], d[temp[0]]
        cnt += 1
        if cnt == k:
            print(*nums)
            exit()

print(-1) 