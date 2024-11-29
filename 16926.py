# Created on iPad.
n, m, r = map(int, input().split())
arr =[]
for _ in range(n):
    arr.append(list(map(int, input().split())))


def rotate(n, m, arr, l): #counter-clockwise
    start = arr[l][l]
    for i in range(l, m-1-l):
        arr[l][i] = arr[l][i+1] #up

    for i in range(l, n-1-l):
        arr[i][m-1-l] = arr[i+1][m-1-l] #right
    
    for i in range(m-1-l, l, -1):
        arr[n-1-l][i] = arr[n-1-l][i-1] #down
    
    for i in range(n-1-l, l, -1):
        arr[i][l] = arr[i-1][l] #left
    
    arr[l+1][l] = start


def main():
    T = min(n, m)//2
    for t in range(T):
        layer = 2*(m-2*t) + 2*(n-2-2*t)
        R = r%layer
        for _ in range(R):
            rotate(n, m, arr, t)

main()
for row in arr:
    print(*row)
