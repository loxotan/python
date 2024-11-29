import sys
input = sys.stdin.readline
from collections import deque
T = int(input())

# BFS로 탐색
def bfs(graph, start, h, w):
    dist = [[-1] * (w+2) for _ in range(h+2)]  # 거리를 저장
    q = deque([start])
    dist[start[0]][start[1]] = 0
    
    # 상하좌우 네 방향
    pos = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    while q:
        x, y = q.popleft()
        
        for dx, dy in pos:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h+2 and 0 <= ny < w+2 and dist[nx][ny] == -1:  # 아직 방문하지 않았을 때
                if graph[nx][ny] == '.':  # 길일 때
                    dist[nx][ny] = dist[x][y]
                    q.appendleft((nx, ny))  # 우선적으로 탐색
                elif graph[nx][ny] == '#':  # 문일 때
                    dist[nx][ny] = dist[x][y] + 1  # 문을 통과할 때는 비용이 증가
                    q.append((nx, ny))  # 나중에 탐색
    
    return dist

def main():
    h, w = map(int, input().split())
    graph = [list('.'*(w+2))]
    for i in range(h):
        graph.append(list('.'+input().strip()+'.'))
    graph.append(list('.'*(w+2)))
    
    prisoner = []
    for i in range(h+2):
        for j in range(w+2):
            if graph[i][j] == '$':
                prisoner.append((i,j))
                graph[i][j] = '.'
    
    
    p1, p2 = prisoner
    dist_p1 = bfs(graph, p1, h, w)
    dist_p2 = bfs(graph, p2, h, w)
    dist_sang = bfs(graph, (0,0), h, w)
    
    result = float('inf')
    for i in range(h):
        for j in range(w):
            if dist_p1[i][j] != -1 and dist_p2[i][j] != -1 and dist_sang[i][j] != -1:
                total_cost = dist_p1[i][j]+dist_p2[i][j]+dist_sang[i][j]
                if graph[i][j] == '#':
                    total_cost -= 2
                result = min(result, total_cost)
    
    return result


for _ in range(T):
    print(main())