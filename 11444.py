n = int(input())

MOD = 10**9+7
def multiply_matrix(a, b):
    return [
        [
            (a[0][0]*b[0][0] + a[0][1]*b[1][0])%MOD,
            (a[0][0]*b[0][1] + a[0][1]*b[1][1])%MOD
        ],
        [
            (a[1][0]*b[0][0] + a[1][1]*b[1][0])%MOD,
            (a[1][0]*b[0][1] + a[1][1]*b[1][1])%MOD
        ]
    ]

def matrix_power(matrix, n):
    result = [[1,0],[0,1]]
    while n>0:
        if n%2 == 1:
            result = multiply_matrix(result, matrix)
        matrix = multiply_matrix(matrix, matrix)
        n //= 2
    return result

def fibonacci(n):
    if n==0:
        return 0
    base_matrix = [[1,1],[1,0]]
    result_matrix = matrix_power(base_matrix, n-1)
    return result_matrix[0][0]

print(fibonacci(n)%MOD)