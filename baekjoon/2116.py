import sys  
input = sys.stdin.readline

T = int(input())
dices = [list(map(int, input().split())) for _ in range(T)]

opp = [5, 3, 4, 1, 2, 0]

ans = 0
for i in range(6):
    bottom = dices[0][i]
    top = dices[0][opp[i]]
    max_side = max(dices[0][j] for j in range(6) if j != i and j != opp[i])
    for j in range(1, T):
        bottom = dices[j][dices[j].index(top)]
        top = dices[j][opp[dices[j].index(bottom)]]
        max_side += max(dices[j][k] for k in range(6) if k != dices[j].index(bottom) and k != dices[j].index(top))
    ans = max(ans, max_side)

print(ans)