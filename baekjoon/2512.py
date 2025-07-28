n = int(input())
want = list(map(int, input().split()))
yesan = int(input())

start, end = 1, max(want)

while start <= end:
    mid = (start+end)//2
    
    best = 0
    for i in want:
        if i-mid >= 0:
            best += mid
        else:
            best += i
    
    if best <= yesan:
        start = mid + 1
    else:
        end = mid - 1
        
print(end)