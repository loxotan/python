N = int(input())

def draw(n, x, y):
    if n == 3:
        graph[y][x] = '*'
        graph[y+1][x-1] = '*'
        graph[y+1][x+1] = '*'
        for i in range(5):
            graph[y+2][x-2+i] = '*'
    else:
        draw(n//2, x, y)
        draw(n//2, x-n//2, y+n//2)
        draw(n//2, x+n//2, y+n//2)

graph = [[' ' for _ in range(2*N-1)] for _ in range(N)]
draw(N, N-1, 0)
print('\n'.join(''.join(row) for row in graph))