import sys
import math
input = sys.stdin.readline

n, s = map(int, input().split())
A = list(map(int, input().split()))
A.append(s)
A.sort()
B = []
for i in range(len(A)-1):
    B.append(A[i+1] - A[i])

while len(B) != 1:
    a = B.pop()
    b = B.pop()
    B.append(math.gcd(a,b))

print(*B)