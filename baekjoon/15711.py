import sys
input = sys.stdin.readline

T = int(input())
lst = []
for i in range(T):
    a, b = map(int, input().split())
    lst.append(a+b)
    
k = max(lst)
nums = [False, False] + [True]*(k-1)
for i in range(2, k+1):
    if nums[i]:
        for j in range(i*i, k+1, i):
            nums[j] = False
            
for i in range(T):
    for j in range(lst[i]):
        if nums[j] and nums[lst[i]-j]:
            print("YES")
            break
    else:
        print("NO")