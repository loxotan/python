import sys
input = sys.stdin.readline

N = int(input())
stack = set()
ans = 0
for _ in range(N):
    name = str(input().rstrip())
    if name == 'ENTER':
        ans += len(stack)
        stack.clear()
    else:
        stack.add(name)
ans += len(stack)
print(ans)