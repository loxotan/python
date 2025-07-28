import sys
input = sys.stdin.readline
import heapq

n = int(input())
left = []
right = []

for _ in range(n):
    num = int(input())
    if len(left) == len(right):
        heapq.heappush(left, -num)
    else:
        heapq.heappush(right, num)
    
    if right and -left[0] > right[0]:
        max_left = -heapq.heappop(left)
        min_right = heapq.heappop(right)
        
        heapq.heappush(left, -min_right)
        heapq.heappush(right, max_left)
    
    print(-left[0])
