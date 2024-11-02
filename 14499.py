import sys
input = sys.stdin.readline

n, m, x, y, k = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(n)]
cmds = list(map(int, input().split()))
dice = [0] * 7

# 각 방향에 따라 주사위가 회전하는 방식 정의
mov = {
    1: [0, 4, 2, 1, 6, 5, 3],  # 동
    2: [0, 3, 2, 6, 1, 5, 4],  # 서
    3: [0, 5, 1, 3, 4, 6, 2],  # 북
    4: [0, 2, 6, 3, 4, 1, 5]   # 남
}

# 명령어에 따른 이동 좌표
directions = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]

def move_dice(dice, cmd):
    new_dice = [0] * 7
    for i in range(1, 7):
        new_dice[i] = dice[mov[cmd][i]]
    return new_dice

def main():
    global x, y
    for cmd in cmds:
        nx, ny = x + directions[cmd][0], y + directions[cmd][1]
        if 0 <= nx < n and 0 <= ny < m:
            x, y = nx, ny
            dice[:] = move_dice(dice, cmd)

            # 바닥면 복사
            if arr[x][y] == 0:
                arr[x][y] = dice[6]
            else:
                dice[6] = arr[x][y]
                arr[x][y] = 0

            # 윗면 출력
            print(dice[1])

main()
