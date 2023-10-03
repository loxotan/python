sudoku = [list(map(int, input().split())) for _ in range(9)]

zero = sum([sudoku[i].count(0) for i in range(9)])

def row():
    if sudoku[i].count(0) != 1:
        return
    else:
        sudoku[i][j] = 45 - sum(sudoku[i])

def col():
    temsum = 0
    temcnt = 0
    for k in range(9):
        temsum += sudoku[k][j]
        if sudoku[k][j] == 0:
            temcnt += 1
    if temcnt != 1:
        return
    else:
        sudoku[i][j] = 45 - temsum
        
def box():
    x = list(range(3*(i//3), 3*(i//3)+3))
    y = list(range(3*(j//3), 3*(j//3)+3))
    temsum = 0
    temcnt = 0
    for a in x:
        for b in y:
            temsum += sudoku[a][b]
            if sudoku[a][b] == 0:
                temcnt += 1
    if temcnt != 1:
        return
    else:
        sudoku[i][j] = 45 - temsum

while zero != 0:
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                row()
                col()
                box()
    zero = sum([sudoku[i].count(0) for i in range(9)])

for k in range(9):
    print(*sudoku[k])
