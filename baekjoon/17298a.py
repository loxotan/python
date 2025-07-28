import sys
input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))
stack = []
ans = [-1]*N

for i in range(N):
    if not stack:
        stack.append(i)
    else:
        while stack and A[stack[-1]] < A[i]:
            ans[stack.pop()] = A[i]
        stack.append(i)
        
print(*ans)