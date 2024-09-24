import sys
input = sys.stdin.readline

n = int(input())
shirts = list(map(int, input().split()))
t, p = map(int, input().split())

ans = 0
for i in range(6):
    if shirts[i]%t>0:
        ans += shirts[i]//t + 1
    else:
        ans += shirts[i]//t 
        
print(ans)
print(n//p, n%p)