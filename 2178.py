import sys
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [[0]]*(n+1)
for i in range(1, n+1):
    line = str(input())
    for j in range(m):
        graph[i].append(int(line[j]))

# bfs
visited = [[0]*(m+1) for _ in range(n+1)]
visited[1][1] = 1
q = [(1,1,0)]

def bfs(a, b):
	while q:
		i, j, d = q.pop(0)
		inpt = [(1,0), (-1,0), (0,1), (0,-1)]
		if i == a and j == b:
			return d
		for io, jo in inpt:
			if 0<=i+io<n+1 and 0<=j+jo<m+1 and visited[i+io][j+jo] != 1 and graph[i+io][j+jo] == 1:
				visited[i+io][j+jo] = 1
				q. append((i+io, j+jo, d+1))
    
    
print(bfs(n, m))

