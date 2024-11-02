n = int(input())
MOD = 1e6
def multiply_matrix(arr1, arr2):
    return [
        [(arr1[0][0]*arr2[0][0] + arr1[0][1]*arr2[1][0])%MOD,
        (arr1[0][0]*arr2[0][1] + arr1[0][1]*arr2[1][1])%MOD],
        [(arr1[1][0]*arr2[0][0] + arr1[1][1]*arr2[1][0])%MOD,
        (arr1[1][0]*arr2[0][1] + arr1[1][1]*arr2[1][1])%MOD]
    ]

def power_matrix(matrix, n):
    base_arr = [[1,0],[0,1]]

    if n == 0:
        return base_arr
    if n == 1:
        return matrix
    
    half = power_matrix(matrix, n//2)
    half = multiply_matrix(half, half)
    
    if n%2 == 1:
        half = multiply_matrix(half, matrix)
    
    return half


def calc_fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    matrix = [[1,1], [1,0]]
    result_matrix = power_matrix(matrix, n-1)
    
    return int(result_matrix[0][0])

print(calc_fib(n))