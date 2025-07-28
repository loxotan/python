S1 = str(input().rstrip())
S2 = str(input().rstrip())

ans = 0
#i번째부터 세어서 연속되지 않을 때 까지 세기
d = [[0] * (len(S2)+1) for _ in range(len(S1)+1)]
for i in range(1, len(S1)+1):
    for j in range(1, len(S2)+1):
        if S1[i-1] == S2[j-1]:
            d[i][j] = d[i-1][j-1]+1
            ans = max(d[i][j], ans)
            
print(ans)