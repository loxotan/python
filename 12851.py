from collections import deque

n, k = map(int, input().split())
arr = [-1]*100001
arr[n] = 0

def bfs(arr, n, k):
    q = deque([n])
    cnt = 0
    min_time = -1
    
    while q:
        now = q.popleft()
        
        if now == k:
            if min_time == -1:
                min_time = arr[now]
                cnt = 1
            elif arr[now] == min_time:
                cnt += 1
            continue
        
        for next in (now-1, now+1, now*2):
            if 0<= next < 100001:
                if arr[next] == -1 or arr[next] == arr[now] + 1:
                    arr[next] = arr[now]+1
                    q.append(next)
    
    return min_time, cnt

min_time, cnt = bfs(arr, n, k)
print(min_time)
print(cnt)
