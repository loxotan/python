import sys
input = sys.stdin.readline

N, M = map(int, input().split())
T = int(input())

cut = []
for i in range(T):
    cut.append(list(map(int, input().split())))
cut.sort()

# 가로로 자르는 선(cut[0] == 0) 중 가장 긴 길이
max_h = 0
prev_h = 0
for c in cut:
    if c[0] == 0:
        max_h = max(max_h, c[1]-prev_h)
        prev_h = c[1]

# 세로로 자르는 선(cut[0] == 1) 중 가장 긴 길이
max_w = 0
prev_w = 0
for c in cut:
    if c[0] == 1:
        max_w = max(max_w, c[1]-prev_w)
        prev_w = c[1]

# 마지막 조각 고려
max_h = max(max_h, M - prev_h)
max_w = max(max_w, N - prev_w)

print(max_h * max_w)