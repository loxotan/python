n = list(map(int,input()))
m = len(n)

#d[n] = 해석될 수 있는 가지 수

d = [0]*(m+1)
if n[0] == 0:
    print('0')
else:
    n = [0] + n
    d[0] = d[1] = 1
    for i in range(2, m+1):
        if n[i] > 0:
            d[i] += d[i-1]
        temp = n[i-1]*10 +n[i]
        if temp >=10 and temp<=26:
            d[i] += d[i-2]
            
    print(d[m]%1000000)

# 0 1 2 2 4 6
# 2 5 1 1 
# 2 5 11
# 25 1 1
# 25 11

# 0 1 2 3 5 8
# 1 1 1 1 1
# 1 1 1 11
# 1 1 11 1
# 1 11 1 1
# 1 11 11
# 11 1 1 1
# 11 1 11
# 11 11 1