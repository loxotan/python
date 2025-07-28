import sys
input = sys.stdin.readline

N = int(input())
S = [input().rstrip() for _ in range(N)]
suffix = dict()
for i in range(N):
    for j in range(len(S[i])):
        if S[i][j:] not in suffix:
            suffix[S[i][j:]] = 1
        else:
            suffix[S[i][j:]] += 1

cnt = 0
for value in suffix.values():
    if value %2 == 1:
        cnt += 1
print(cnt)