import sys
input = sys.stdin.readline

from collections import deque

T = int(input())

for __ in range(T):
    a, b = map(int, input().split())
    q = deque([(a, '')]) 
    
    visited = [False]*10000
    visited[a] = True
    
    while q:
        x = q.popleft()
        if x[0] == b:
            print(x[1])
            break
        
        for cmd in ['D', 'S', 'L', 'R']:
            if cmd == 'D':
                nx = (2*x[0]) % 10000
            elif cmd == 'S':
                nx = (x[0]-1) % 10000
            elif cmd == 'L':
                nx = (x[0]%1000)*10 + x[0]//1000
            else: # cmd == 'R'
                nx = (x[0]%10)*1000 + x[0]//10
            
            if not visited[nx]:
                q.append((nx, x[1]+cmd))
                visited[nx] = True
        

    