# SK = 1
# CY = 0

dp = [0] *1001
dp[1] = 1
dp[2] = 0
dp[3] = 1
dp[4] = 1
# 5 = 1 3>#2
# 6 = 1 4>#2
# 7 = 0 
# 8 = 1 1>#7
# 9 = 0

for i in range(5, 1001):
    dp[i] = (min(dp[i-1], dp[i-3], dp[i-4]) +1)%2
    
N = int(input())
if dp[N] == 0:
    print('CY')
else:
    print('SK')
