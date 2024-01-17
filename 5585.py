N = 1000 - int(input())
coins = [500, 100, 50, 10, 5, 1]

ans = 0
for coin in coins:
    ans += N//coin
    N %= coin
print(ans)