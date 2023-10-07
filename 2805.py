n, m = map(int, input().split())
trees = list(map(int, input().split()))

high = max(trees)
low = 1

while low <= high:
    mid = (high+low)//2
    
    amount = 0
    for tree in trees:
        if tree-mid >= 0:
            amount += tree - mid
            
    if amount>=m:
        low = mid+1
    else:
        high = mid-1

print(high)