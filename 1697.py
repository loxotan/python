from collections import deque

n, k = map(int, input().split())
chk = [0]*100001

q = deque([])
def bfs():
    while q:
        v  = q.popleft()
        if v == k:
            print(chk[v])
            break
        for i in (v-1, v+1, v*2):
            if 0<= i <= 100000 and chk[i] == 0:
                chk[i] = chk[v] + 1
                q.append(i)
            

q.append(n)
bfs()