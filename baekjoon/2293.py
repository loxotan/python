n, k = map(int, input().split())
coins = [int(input()) for _ in range(n)]

#i번째 동전을 1~i-1번째 동전으로 몇개나 만들수 잇는지
price = [1]
for i in range(1, n):
    tem_price = 1
    c = coins[i]
    for j in range(i):
        tem_price += price[j] * c//coins[j]
        c = c%coins[j]
    if c != 0:
        tem_price = 1
    price.append(tem_price)
    
coins.sort(reverse = True)
how_many = 0
for i in range(n):
    how_many += price[i] * k//coins[i]
    k = k%coins[i]
    
print(how_many)