import sys
n = int(input())
a = int(sys.stdin.readline())

garosu = []
for i in range(n-1):
    num = int(sys.stdin.readline())
    garosu.append(num-a)
    a = num
    
import math
gcd = math.gcd(*garosu) 

result = 0
for each in garosu:
    result += each//gcd-1
print(result)