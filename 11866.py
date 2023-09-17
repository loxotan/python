n, k = map(int, input().split())
from collections import deque
list = deque([i+1 for i in range(n)])
josep = []

while len(list)>0:
    list.rotate(-k)
    josep.append(list.pop())
    
print('<', end='')    
print(*josep, sep=', ', end='')
print('>')
