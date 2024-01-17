import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
graph = [[] for _ in range(n)]
for i in range(n):
    line = input()
    for j in range(m):
        graph[i].append(int(line[j]))

# bfs
visited = [[0]*m for _ in range(n)]
visited[0][0] = 1
q = deque()
q.append([0,0,0])

def bfs(a, b):
	while q:
		i, j, d = q.popleft()
		inpt = [(1,0), (-1,0), (0,1), (0,-1)]
		if i == a and j == b:
			return d + 1
		for io, jo in inpt:
			if 0<=i+io<n and 0<=j+jo<m and not visited[i+io][j+jo] and graph[i+io][j+jo]:
				visited[i+io][j+jo] = 1
				q. append([i+io, j+jo, d+1])
    
    
print(bfs(n-1, m-1))

