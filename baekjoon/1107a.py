import sys
input = sys.stdin.readline

N = int(input())
M = int(input())

broken = [False] * 10
if M > 0:
    a = list(map(int, input().split()))
else:
    a = []
for x in a:
    broken[x] = True

def check(c):
    if c == 0:
        if broken[0]:
            return 0
        else:
            return 1
        
    l = 0
    while c>0:
        if broken[c%10]:
            return 0
        l += 1
        c//= 10
    return l

ans = abs(N-100)
for i in range(0, 1000000+1):
    c = i
    l = check(c)
    if l > 0:
        press = abs(c-N)
        if ans > l + press:
            ans = l + press

print(ans)