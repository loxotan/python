n, m = map(int, input().split())

def funct(a, b, c):
    cnt = 0
    for i in range(a+1):
        for j in range(b+1):
            if i + j == c:
                cnt += 1
    return cnt

hap = 0
for k in range(n+m+1):
    hap += funct(n, m, k)
    
print(hap)