import sys
input = sys.stdin.readline

m, n = map(int, input().split())
tomato = [[] for _ in range(n)]
for i in range(n):
    tomato[i] = list(map(int, input().split()))
    
#bfs를 한 단계씩 시행해서 동시에 모든 토마토에서 하루씩 익어갈 수 있게

def bfs(a, b): #(a,b)에서 d일째 시행
    global keep_going
    for i, j in [(1,0), (-1,0), (0, 1), (0,-1)]:
        if 0<=a+i<n and 0<=b+j<m and tomato[a+i][b+j] == 0:
            tomato[a+i][b+j] = 1 #주변 토마토 익힘

days = [tomato]
while True:
    for i in range(n):
        for j in range(m):
            if tomato[i][j] == 1:
                bfs(i, j)
    days.append(tomato)    
    if days[-1] == days[-2]:
        break

print(len(days))
    