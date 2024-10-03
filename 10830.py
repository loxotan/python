n, b = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(n)]

def mult_mat(arr1, arr2):
    res = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                res[i][j] += arr1[i][k]*arr2[k][j]
    
    for i in range(n):
        for j in range(n):
            res[i][j] %= 1000

    return res

def calc(arr, b):
    if b == 1:
        return [[element % 1000 for element in row] for row in arr]
    elif b % 2 == 0:
        half = calc(arr, b // 2)
        return mult_mat(half, half)
    else:
        return mult_mat(calc(arr, b - 1), arr)
    
ans = calc(arr, b)
for row in ans:
    print(*row)