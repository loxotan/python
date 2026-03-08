import sys 
input = sys.stdin.readline

T = int(input())
cars = list(map(int, input().split()))
m = int(input())

# p[i] = cars[0]+cars[1]+...+cars[i-1]
p = [0] * (T+1)
for i in range(1, T+1):
    p[i] = p[i-1] + cars[i-1]

dp= [[0] * (T+1) for _ in range(4)]
for k in range(1, 4):
    for i in range(k*m, T+1):
        dp[k][i] = max(dp[k][i-1], dp[k-1][i-m] + (p[i]-p[i-m]))

print(dp[3][T])