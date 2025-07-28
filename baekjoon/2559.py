import sys
input = sys.stdin.readline

n, k = map(int, input().split())
X = list(map(int, input().split()))
plus = [0]

temp = 0
for i in X:
    temp += i
    plus.append(temp)

sums = []
for i in range(k, n+1):
    sums.append(plus[i] - plus[i-k])
    
print(max(sums))
    