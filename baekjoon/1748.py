N = int(input())

ans = 0
S = [0, 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999]
for s in range(1, len(S)):
    if N <= S[s]:
        ans += (N-S[s-1])*s
        break
    else:
        ans += (s)*(S[s]-S[s-1])
        
print(ans)