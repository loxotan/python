import sys
input = sys.stdin.readline

T = int(input())
dungchi = []
for _ in range(T):
    dungchi.append(list(map(int, input().split())))

sunseo = []
for i in range(T):
    cnt = 1
    for j in range(T):
        if dungchi[i][0]<dungchi[j][0] and dungchi[i][1]<dungchi[j][1]:
            cnt += 1
    sunseo.append(cnt)

print(*sunseo, sep=' ')   