T = int(input())

for _ in range(T):
    N = int(input())
    coins = list(map(int, input().split()))
    M = int(input())
    
    dp = [0] * 10001
    
    for i in range(1, 10001):
        for coin in coins:
            if i+coin < 10001:
                dp[i+coin] = dp[i] + 1
    
    print(dp[M])
    
