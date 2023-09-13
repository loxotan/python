import sys
n = int(input())
a1 = int(sys.stdin.readline())

garosu = []
for i in range(n-1):
    garosu.append(int(sys.stdin.readline())-a1)
    
import math
gcd = math.gcd(*garosu) 

garosu_list = []
for j in range(len(garosu)):
    garosu_list.append(garosu[j]//gcd)
    
print(len(set(range(garosu_list[-1]))-set(garosu_list))-1)