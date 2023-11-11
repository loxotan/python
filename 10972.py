import sys
input = sys.stdin.readline

N = int(input())
ans = list(map(int, input().split()))

for i in range(N-1, 0, -1):
    if ans[i-1] < ans[i]:
        for j in range(N-1, 0, -1):
            if ans[i-1] < ans[j]:
                ans[i-1], ans[j] = ans[j], ans[i-1]
                ans = ans[:i] + sorted(ans[i:])
                for i in ans:
                    print(i, end = ' ')
                exit()
print(-1)