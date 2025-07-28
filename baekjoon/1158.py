from collections import deque

n, k = map(int, input().split())

q = deque(list(_ for _ in range(1, n+1)))
josep = []
while q:
    q.rotate(-k+1)
    x = q.popleft()
    josep.append(x)
    
print('<', end = '')
print(*josep, sep = ', ', end = '')
print('>')