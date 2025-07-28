import sys
input = sys.stdin.readline

upper = []
lower = []
num = [str(i) for i in range(10)]
for i in range(26):
    upper.append(chr(i+65))
    lower.append(chr(i+97))

while True:
    ans = [0, 0, 0, 0]
    S = input().rstrip('\n')
    if len(S) == 0:
        break
    
    for x in S:
        if x in lower:
            ans[0] += 1
        elif x in upper:
            ans[1] += 1
        elif x in num:
            ans[2] += 1
        else: #공백
            ans[3] += 1
    
    print(*ans)
