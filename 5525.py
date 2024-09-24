

n = int(input())
m = int(input())
s = input().rstrip()

#50%
"""pn = 'I'+'OI'*n
ans = 0
for i in range(m):
    if s[i:i+2*n+1] == pn:
        ans += 1

print(ans)"""

#retry
ans, i, count = 0, 0, 0

while i<(m-1):
    if s[i:i+3] =='IOI':
        i+=2
        count += 1
        if count == n:
            ans += 1
            count -= 1
    else:
        i+=1
        count = 0
        
print(ans)
