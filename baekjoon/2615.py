import sys
input = sys.stdin.readline

board = [list(map(int, input().split())) for _ in range(19)]

def check(x, y):
    color = board[x][y]
    d1 = True
    d2 = True
    h = True
    v = True
    for i in range(5):
        if x+i >= 19 or y+i >= 19 or board[x+i][y+i] != color: # diagnal
            d1 = False

    for i in range(5):
        if x-i < 0 or y+i >= 19 or board[x-i][y+i] != color: # anti-diagonal
            d2 = False
    
    for i in range(5):
        if x+i >= 19 or board[x+i][y] != color: # vertical
            v = False

    for i in range(5):
        if y+i >= 19 or board[x][y+i] != color: # horizontal
            h = False

    if d1:
        if x-1 >= 0 and y-1 >= 0 and board[x-1][y-1] == color: # 6목 이상
            return False
        if x+5 < 19 and y+5 < 19 and board[x+5][y+5] == color: # 6목 이상
            return False
        return True
    if d2:
        if x+1 < 19 and y-1 >= 0 and board[x+1][y-1] == color: # 6목 이상
            return False
        if x-5 >= 0 and y+5 < 19 and board[x-5][y+5] == color: # 6목 이상
            return False
        return True
    if v:
        if x-1 >= 0 and board[x-1][y] == color: # 6목 이상
            return False
        if x+5 < 19 and board[x+5][y] == color: # 6목 이상
            return False
        return True
    if h:
        if y-1 >= 0 and board[x][y-1] == color: # 6목 이상
            return False
        if y+5 < 19 and board[x][y+5] == color: # 6목 이상
            return False
        return True
    return False

for i in range(19):
    for j in range(19):
        if board[i][j] != 0 and check(i, j):
            print(board[i][j])
            print(i+1, j+1)
            exit(0)

print(0)