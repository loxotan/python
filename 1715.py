import sys
import heapq
input = sys.stdin.readline

N = int(input())
heap = list()

for i in range(N):
    heapq.heappush(heap, int(input()))

cnt = 0
while len(heap) > 1:
    x = heapq.heappop(heap)
    y = heapq.heappop(heap)
    cnt += x+y
    heapq.heappush(heap, x+y)

print(cnt)