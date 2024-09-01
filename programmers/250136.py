#각 열마다 행을 따라 내려가며 BFS
#여러 열에 걸쳐 있을 경우 열 list 에 + BFS 최종 값
## BFS를 하면서 걸리는 열을 기록했다가 마지막에 열 list에 더해주기, False로 바꿔서 다시 지나가지 않게
#max(answer)
#각 열마다 행을 따라 내려가며 BFS
#여러 열에 걸쳐 있을 경우 열 list 에 + BFS 최종 값
## BFS를 하면서 걸리는 열을 기록했다가 마지막에 열 list에 더해주기, False로 바꿔서 다시 지나가지 않게
#max(answer)
from collections import deque

def solution(land):
    answer = [0] * len(land[0])
    
    def bfs(start_row, start_col, land):
        queue = deque([(start_row, start_col)])
        land[start_row][start_col] = 0
        cols_visited = set([start_col])
        depth = 1
        
        moves = [(1,0), (0,1)]
        
        while queue:
            row, col = queue.popleft()
            for move in moves:
                new_row, new_col = row+move[0], col+move[1]
                if 0<= new_row < len(land) and 0<= new_col < len(land[0]) and land[new_row][new_col]:
                    land[new_row][new_col] = 0
                    queue.append((new_row, new_col))
                    cols_visited.add(new_col)
                    depth += 1
                    
        return cols_visited, depth
    
    for i in range(len(land[0])):
        for j in range(len(land)):
            if land[j][i] != 0:
                cols, d = bfs(j, i, land)
                for col in cols:
                    answer[col]+=d
    
    return max(answer)

def main():
    land = [[1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    return solution(land)

print(main())