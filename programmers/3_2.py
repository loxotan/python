def solution(rows, cols, queries):
    arr = [[(i) * cols + (j+1) for j in range(cols)] for i in range(rows)]
    answer = []
    for query in queries:
        x1, y1, x2, y2 = query[0]-1, query[1]-1, query[2]-1, query[3]-1
        row1, row2 = arr[x1][y1:y2], arr[x2][y1+1:y2+1]
        _min = min(row1+row2)
        
        for i in range(x1, x2):
            arr[i][y1] = arr[i+1][y1]
            if arr[i][y1] < _min : _min = arr[i][y1]
        
        for i in range(x2, x1, -1):
            arr[i][y2] = arr[i-1][y2]
            if arr[i][y2] < _min : _min = arr[i][y2]
        
        arr[x1][y1+1:y2+1], arr[x2][y1:y2] = row1, row2
        
        answer.append(_min)
    return answer

solution(6, 6, [[2, 2, 5, 4],[3,3,6,6],[5,1,6,3]])