import sys
input = sys.stdin.readline
from collections import deque

n = int(input())
arr = []
for _ in range(n):
    arr.append(list(map(int, input().split())))

ans = [[0]*n for _ in range(n)]

def reach(arr, start):
    q = deque([(start)])
    visited = [False]*n
    
    while q:
        next = q.popleft()

        for i in range(n):
            if arr[next][i] == 1 and not visited[i]:
                visited[i] = True
                ans[start][i] = 1
                q.append(i)

for i in range(n):
    reach(arr, i)

for row in ans:
    print(*row)