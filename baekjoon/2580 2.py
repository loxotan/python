import sys
input = sys.stdin.readline

sudoku = []
blank = []
for i in range(9):
    sudoku.append(list(map(int, input().split())))
    
for i in range(9):
    for j in range(9):
        if sudoku[i][j] == 0:
            blank.append([i,j])
            
def row(x, a):
    for i in range(9):
        if a == sudoku[x][i]:
            return False
    return True

def col(y, a):
    for i in range(9):
        if a == sudoku[i][y]:
            return False
    return True

def box(x, y, a):
    nx = x//3 * 3
    ny = y//3 * 3
    for i in range(3):
        for j in range(3):
            if a == sudoku[nx+i][ny+j]:
                return False
    return True

cnt = 0
def dfs(idx):
    global cnt
    if idx == len(blank):
        for i in range(9):
            print(*sudoku[i])
        print(cnt)
        exit(0)
    
    for i in range(1, 10):
        x = blank[idx][0]
        y = blank[idx][1]
        
        if row(x, i) and col(y, i) and box(x, y, i):
            sudoku[x][y] = i
            cnt += 1
            dfs(idx+1)
            sudoku[x][y] = 0
            
dfs(0)