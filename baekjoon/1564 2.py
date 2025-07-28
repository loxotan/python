import sys
input = sys.stdin.readline

k, n = map(int, input().split())
lan = []
for _ in range(k):
    lan.append(int(input()))

start, end = 1, sorted(lan)[-1]


while start<=end:
    mid = (start+end)//2
    cnt = 0
    for i in lan:
        cnt += i//mid
    if cnt<n:
        end = mid-1
    else:
        start = mid+1
        
print(end)