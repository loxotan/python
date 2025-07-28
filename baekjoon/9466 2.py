import sys
input = sys.stdin.readline

T = int(input())

def dfs(graph, visited, team, start):
    path = []
    cur = start
    
    while True:
        path.append(cur)
        visited[cur] = True
        next = graph[cur]
        
        if visited[next]:
            if next in path:
                cycle_start = path.index(next)
                for i in range(cycle_start, len(path)):
                    team[path[i]] = True
            break
        cur = next
        
    for node in path:
        visited[node] = True

def main():
    for _ in range(T):
        n = int(input())
        graph = [0] + list(map(int, input().split()))
        ans = n
        visited = [False] * (n+1)
        team = [False] * (n+1)

        for i in range(1, n+1):
            if not visited[i]:
                dfs(graph, visited, team, i)
        
        ans = n - sum(team)
        print(ans)


main()
    