import sys
input = sys.stdin.readline

n = int(input())
arr = []
for _ in range(n):
    arr.append(list(map(int, input().split())))

def is_same_color(arr, x, y, n):
    color = arr[x][y]
    for i in range(x, x+n):
        for j in range(y, y+n):
            if arr[i][j] != color:
                return False
    return True

def dnc(arr, x, y, n, result):
    if is_same_color(arr, x, y, n):
        if arr[x][y] == 0:
            result['white'] += 1
        else:
            result['blue'] += 1
        return
    
    mid = n//2
    dnc(arr, x, y, mid, result)
    dnc(arr, x, y+mid, mid, result)
    dnc(arr, x+mid, y, mid, result)
    dnc(arr, x+mid, y+mid, mid, result)

def solve(arr, n):
    result = {'white':0, 'blue':0}
    dnc(arr, 0, 0, n, result)
    print(result['white'])
    print(result['blue'])
    
solve(arr, n)