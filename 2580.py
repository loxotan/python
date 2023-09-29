sudoku = [list(map(int, input().split())) for _ in range(9)]

zero = sum([graph[i].count(0) for i in range(9)])

def row():
    if graph[i].count(0) != 1:
        return
    else:
        graph[i][j] = 81 - sum(graph[i])

def box():
    x = range(3*(i//3), 3*(i//3)+3)
    y = range(3*(j//3), 3*(j//3)+3)
    temsum = 0
    temcnt = 0
    for a in x:
        for b in y:
            temsum += graph[a][b]
            if graph[a][b] == 0:
                temcnt += 1
    if temcnt != 1:
        return
    else:
        graph[i][j] = 81 - temsum

while zero != 0:
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                row()
                box()
    zero = sum([graph[i].count(0) for i in range(9)])

for k in range(9):
    print(*graph[k])
