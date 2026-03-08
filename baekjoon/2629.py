import sys
input = sys.stdin.readline

T = int(input())
weights = list(map(int, input().split()))
weights.sort()

K = int(input())
balls = list(map(int, input().split()))

S = sum(weights) # 추로 만들 수 있는 최대 무게
possible = [False]*(S+1)
possible[0] = True

for w in weights:
    nxt = possible[:]          # 안 쓰는 경우 포함
    for d in range(S+1):
        if possible[d]:
            if d + w <= S: nxt[d+w] = True
            nxt[abs(d-w)] = True
    possible = nxt

for b in balls:
    if b > S: # 추로 만들 수 있는 최대 무게보다 무거운 경우
        print('N', end=' ')
    elif possible[b]: # b 무게를 만들 수 있는 경우
        print('Y', end=' ')
    else: # b 무게를 만들 수 없는 경우
        print('N', end=' ')