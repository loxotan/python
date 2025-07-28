#  0
#     1        2        3       4          5         6        7         8        9 
#    0  2    1  3     2  4     3  5      4  6      5  7      6  8     7  9      8
#   1   13  02  24   13  35   24  46    35  57    46  68    57  79    68  8    79
#  02  024 113 1335 024 1246 1335 3557 2446 4668 3557 5779 4668 688 5779  79  688
# 
# 1 9 17 32 61 115

n = int(input())
suyeol = {0:1, 1:9}
gyedan = [1,2,3,4,5,6,7,8,9]
for i in range(2, n+1):
    gyedan_next = []
    for int in gyedan:
        if int == 1:
            gyedan_next.extend([0,2])
        elif int == 2:
            gyedan_next.extend([1,3])
        elif int == 3:
            gyedan_next.extend([2,4])
        elif int == 4:
            gyedan_next.extend([3,5])
        elif int == 5:
            gyedan_next.extend([4,6])
        elif int == 6:
            gyedan_next.extend([5,7])
        elif int == 7:
            gyedan_next.extend([6,8])
        elif int == 8:
            gyedan_next.extend([7,9])
        elif int == 9:
            gyedan_next.extend([8])
        elif int == 0:
            gyedan_next.extend([1])
    gyedan = gyedan_next
    suyeol[i] = len(gyedan) 

suyeol_2 = {0:1, 1:9}
for i in range(2, n+1):
    suyeol_2[i] = 2*suyeol_2[i-1] - (i-1)

print(suyeol[n])
print(suyeol_2[n])

dp = [ [0]*10 for _ in range(n+1)]

for i in range(1,10):
    dp[1][i] = 1
    
for i in range(2, n+1):
    for j in range(10):
        if j == 0:
            dp[i][j] = dp[i-1][1]
        elif j == 9:
            dp[i][j] = dp[i-1][8]
        else:
            dp[i][j] = dp[i-1][j-1] + dp[i-1][j+1]
            
print((sum(dp[n]))%1000000000)
