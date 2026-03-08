import sys 
input = sys.stdin.readline

N, M = map(int, input().split())
ladder = []
snake = []
for _ in range(N):
    a, b = map(int, input().split())
    ladder.append((a, b))
for _ in range(M):
    a, b = map(int, input().split())
    snake.append((a, b))

from collections import deque
def dfs():
    visited = [False] * 101
    q = deque([(1, 0)])  # (현재 위치, 주사위 던진 횟수)
    visited[1] = True

    while q:
        pos, cnt = q.popleft()
        if pos == 100:
            return cnt
        
        for dice in range(1, 7):
            next_pos = pos + dice
            if next_pos > 100:
                continue
            
            # 사다리나 뱀을 타고 이동
            for a, b in ladder:
                if next_pos == a:
                    next_pos = b
                    break
            for a, b in snake:
                if next_pos == a:
                    next_pos = b
                    break
            
            if not visited[next_pos]:
                visited[next_pos] = True
                q.append((next_pos, cnt + 1))

print(dfs())