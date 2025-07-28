import sys
input = sys.stdin.readline

m, n, inv = map(int, input().split())
mine = [list(map(int, input().split())) for _ in range(m)]

#제일 낮은 층, 높은 층 구하기
max_mine = max(list(max(mine[_]) for _ in range(m)))
min_mine = min(list(min(mine[_]) for _ in range(m)))

#제일 낮은 층으로 만들 때 걸리는 시간부터 제일 높은 층으로 만들 때 걸리는 시간 구하기
time_list = []
for h in range(min_mine, max_mine+1):
    time = 0
    temp_inv = 0
    temp_need = 0
    for i in range(m):
        for j in range(n):
            if mine[i][j] > h: #도달하려는 높이보다 땅이 높을 경우
                temp_inv += mine[i][j] - h #깎은 양 만큼 임시저장
                time += 2*(mine[i][j] - h )
            elif mine[i][j] < h: #도달하려는 높이보다 땅이 낮은 경우
                temp_need += h - mine[i][j] #필요한 양 만큼 임시저장
                time += h - mine[i][j]
    if inv + temp_inv >= temp_need:
        time_list.append([time, h])
    else:
        break

time_final = sorted(time_list, key = lambda x: (x[0], -x[1]))
print(*time_final[0], sep =' ')

