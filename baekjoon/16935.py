import sys
input = sys.stdin.readline

n, m, r = map(int, input().split())
arr = []
for _ in range(n):
    arr.append(list(map(int, input().split())))

def one(n, m, arr):
    return arr[::-1]

def two(n, m, arr):
    arr2 = [row[::-1] for row in arr]
    return arr2

def three(n, m, arr):
    arr2 = [[] for _ in range(m)]
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            arr2[j].append(arr[i][j])
    return arr2

def four(n, m, arr):
    arr2 = [[] for _ in range(m)]
    for i in range(n):
        for j in range(m-1, -1, -1):
            arr2[j].append(arr[i][j])
    arr3 = one(m, n, arr2)
    return arr3

def five(n, m, arr):
    mid_row, mid_col = n//2, m//2
    arr2 = []
    for i in range(mid_row):
        arr2.append(arr[i+mid_row][:mid_col]+arr[i][:mid_col]) #q4+q1
    for i in range(mid_row, n):
        arr2.append(arr[i][mid_col:]+arr[i-mid_row][mid_col:]) #q3+q2
    return arr2

def six(n, m, arr):
    mid_row, mid_col = n//2, m//2
    arr2 = []
    for i in range(mid_row):
        arr2.append(arr[i][mid_col:]+arr[i+mid_row][mid_col:]) #q2+q3
    for i in range(mid_row, n):
        arr2.append(arr[i-mid_row][:mid_col]+arr[i][:mid_col]) #q1+q4
    return arr2

def main():
    cmds = list(map(int, input().split()))
    global n, m, arr
    for cmd in cmds:
        if cmd == 1:
            arr = one(n, m, arr)
        elif cmd == 2:
            arr = two(n, m, arr)
        elif cmd == 3:
            arr = three(n, m, arr)
            n, m = m, n
        elif cmd == 4:
            arr = four(n, m, arr)
            n, m = m, n
        elif cmd == 5:
            arr = five(n, m, arr)
        elif cmd == 6:
            arr = six(n, m, arr)
    return arr

arr = main()
for row in arr:
    print(*row)