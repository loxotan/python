import sys
input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))
setA = list(set(A))
dic = {}

for i in range(len(setA)):
    
