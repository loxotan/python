import sys 
input = sys.stdin.readline
import heapq

T = int(input())
for _ in range(T):
    N = int(input())
    if N == 0:
        print("EMPTY")
        continue
    else:
        min_heap = []
        max_heap = []
        visited = [False] * N
        for i in range(N):
            cmd, num = input().split()
            num = int(num)
            if cmd == "I":
                heapq.heappush(min_heap, (num, i))
                heapq.heappush(max_heap, (-num, i))
                visited[i] = True
            else:
                if num == 1:
                    while max_heap and not visited[max_heap[0][1]]:
                        heapq.heappop(max_heap)
                    if max_heap:
                        visited[max_heap[0][1]] = False
                        heapq.heappop(max_heap)
                else:
                    while min_heap and not visited[min_heap[0][1]]:
                        heapq.heappop(min_heap)
                    if min_heap:
                        visited[min_heap[0][1]] = False
                        heapq.heappop(min_heap)

        while min_heap and not visited[min_heap[0][1]]:
            heapq.heappop(min_heap)
        while max_heap and not visited[max_heap[0][1]]:
            heapq.heappop(max_heap)

        if not min_heap or not max_heap:
            print("EMPTY")
        else:
            print(-max_heap[0][0], min_heap[0][0])