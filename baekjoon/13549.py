from collections import deque

n, k = map(int, input().split())
arr = [-1]*100001
arr[n] = 0

def bfs(arr, n, k):
    q = deque([n])
    
    while q:
        now = q.popleft()
        
        if now == k:
            return arr[now]
        
        for next in (now*2, now-1, now+1):
            if 0<= next < 100001 and arr[next] == -1:
                if next == now*2:
                    arr[next] = arr[now]
                    q.appendleft(next)
                elif next == now-1 or next == now+1:
                    arr[next] = arr[now]+1
                    q.append(next)


print(bfs(arr, n, k))

