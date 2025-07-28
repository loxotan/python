n = int(input())
li = [list(map(int, input().split())) for _ in range(n)]
li.sort()

dp = [1]*n

for i in range(1, n):
    for j in range(0, i):
        if li[j][1] < li[i][1]:
            dp[i] = max(dp[i], dp[j] + 1)
        
print(n-max(dp))