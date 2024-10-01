import sys
input = sys.stdin.readline

TC = int(input())


def negative_cycle(graph, n):
    distance = [0] * (n+1)
    
    for i in range(n):
        for u in range(1, n+1):
            for v, weight in graph[u]:
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    if i == n-1:
                        return True

    return False


def main():
    for _ in range(TC):
        n, m, w = map(int, input().split())
        graph = [[] for _ in range(n+1)]
        for _ in range(m):
            s, e, t = map(int, input().split())
            graph[s].append((e, t))
            graph[e].append((s, t))
        for _ in range(w):
            s, e, t = map(int, input().split())
            graph[s].append((e, -t))
        
        if negative_cycle(graph, n):
            print("YES")
        else:
            print("NO")

main()