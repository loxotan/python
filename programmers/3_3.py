def solution(n):
    answer = []
    arr = [[0]*n for _ in range(n+1)]
    move = [[1, 0], [0, 1], [-1, -1]]
    arr[0][0] = 0
    x, y = 0, 0
    for i in range(n, 0, -1):
        for j in range(i):
            arr[x+move[(n-i)%3][0]][y+move[(n-i)%3][1]] = arr[x][y]+1
            x = x+move[(n-i)%3][0]
            y = y+move[(n-i)%3][1]
    
    for line in arr:
        for num in line:
            if num != 0:
                answer.append(num)
    
    return answer

print(solution(4))
print(solution(5))