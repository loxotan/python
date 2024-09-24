import sys
input = sys.stdin.readline

n = int(input())
graph = [int(input()) for _ in range(n)]
arr = [[start, end] for start, end in enumerate(graph, start = 1)]
#첫째줄은 시작, 둘째줄은 끝인 그래프
ans = set()
for i in range(n):
    visited = [False]*n
    start, end = arr[i]
    loop = [end]
    while visited[end-1] == False:
        new_end = graph[end-1]
        loop.append(new_end)
        if start == new_end:
            for num in loop:
                ans.add(num)
            break
        visited[end-1] = True
        end = new_end

ans = sorted(list(ans))
print(len(ans))
for num in ans:
    print(num)