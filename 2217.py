import sys
input = sys.stdin.readline

n = int(input())
ropes = [int(input()) for _ in range(n)]
ropes.sort()

# 10, 100, 10000 이 주어졌을 때 k = 10000
# 1 1 1 10 10 100 이 주어졌을 때 k = 100
# 6, 5, 4, 30, 20, 100

weight = []
for i in range(n):
    weight.append(ropes[i]*(n-i))
    
print(max(weight))