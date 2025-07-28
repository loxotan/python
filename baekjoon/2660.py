import sys
input = sys.stdin.readline

n = int(input())
arr = [[0]*(n+1) for _ in range(n+1)]

a, b = map(int, input().split())
while a != -1 and b != -1:
    arr[a][b] = 1
    arr[b][a] = 1
    a, b = map(int, input().split())
    
def bfs(i, arr):
    visited = [False]*(n+1)
    distance = [0]*(n+1)
    from collections import deque
    q = deque([i])
    visited[i] = True
    
    while q:
        node = q.popleft()
        for j in range(1, n+1):
            if not visited[j] and arr[node][j]:
                distance[j] = distance[node] + 1
                visited[j] = True
                q.append(j)
                
    return max(distance)

ans = [0]*(n+1)
for i in range(1, n+1):
    ans[i] = bfs(i, arr)

head = min(ans[1:])
print(head, ans.count(head))

res =[]
for i in range(1, n+1):
    if ans[i] == head:
        res.append(i)
print(*res)