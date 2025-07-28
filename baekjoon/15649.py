n, m = map(int, input().split())

visited = [False] * (n+1)
answer = []
def btk(start):
    if len(answer) == m:
        print(*answer)
    
    for i in range(start, n+1):
        if not visited[i]:
            answer.append(i)
            visited[i] = True
            btk(i)
            answer.pop()
            visited[i] = False
            
btk(1)