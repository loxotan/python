import sys
input = sys.stdin.readline

n, k = map(int, input().split())
coins = [int(input()) for _ in range(n)]
coins.sort(reverse = True)

how_many = 0
for coin in coins:
    how_many += k//coin
    k = k%coin
    
print(how_many)