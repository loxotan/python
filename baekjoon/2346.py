from collections import deque

N = int(input())
balloon = deque(enumerate(map(int, input().split())))

ans = []

for i in range(N):
    B = balloon.popleft()
    ans.append(B[0]+1)
    if B[1] > 0:
        balloon.rotate(-B[1]+1)
    else:
        balloon.rotate(-B[1])

print(*ans)