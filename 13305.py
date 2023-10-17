import sys
input = sys.stdin.readline

n = int(input())
dists = list(map(int, input().split()))
prices = list(map(int, input().split()))

i = 0
rst = 0
num =1

while (i+num) < n:
    if prices[i] > prices[i+num]:
        rst += prices[i]*sum(dists[i:i+num])
        i += num
        num = 1
    else:
        num += 1
        
else:
    rst += prices[i]*sum(dists[i:])
    
print(rst)