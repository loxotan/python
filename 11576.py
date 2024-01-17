import sys
input = sys.stdin.readline

A, B = map(int, input().split())
N = int(input())
num = list(map(int, input().split()))
ans = 0
res = []
for i in range(N):
    ans += num[-i-1] * (A**i)

while ans:
    res.append(ans%B)
    ans = ans//B

print(*res[::-1])