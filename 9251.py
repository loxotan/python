S1 = str(input().rstrip())
S2 = str(input().rstrip())

dp = [[0]*(len(S2)+1) for _ in range(len(S1)+1)]
#dp[S1의 i번째 글자][S2의 j번째 글자]

for i in range(1, len(S1)+1):
    for j in range(1, len(S2)+1):
        if S1[i-1] == S2[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i][j-1], dp[i-1][j])

print(dp[-1][-1])