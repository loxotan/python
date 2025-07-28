import sys
input = sys.stdin.readline

n = int(input())
switch = [0] + list(map(int, input().split())) #switch[n] = n번째 스위치

m = int(input())
for _ in range(m):
    student = list(map(int, input().split()))
    if student[0] == 1: #남학생
        k = 1
        while k*student[1] <= n:
            switch[k*student[1]] = (switch[k*student[1]]+1)%2
            k += 1
    else: #여학생
        j = 1
        switch[student[1]] = (switch[student[1]]+1)%2
        while 0<student[1]-j and student[1]+j<=n and switch[student[1]-j] == switch[student[1]+j]:
            switch[student[1]-j] = (switch[student[1]-j]+1)%2
            switch[student[1]+j] = (switch[student[1]+j]+1)%2
            j += 1

for i in range(1, n+1, 20):
    print(*switch[i:i+20]) 