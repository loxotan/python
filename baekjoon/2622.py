import sys
input = sys.stdin.readline

T = int(input())

ans = 0
for a in range(1, T // 3 + 1):
    b_max = (T - a) // 2
    b_min = max(a, (T - 2 * a) // 2 + 1)  # b > T/2 - a

    if b_min <= b_max:
        ans += (b_max - b_min + 1)

print(ans)
