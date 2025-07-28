import sys
input = sys.stdin.readline

T = int(input())

for _ in range(T):
    H, W, N = map(int, input().split())
    n1 = (N%H if N%H != 0 else H) #층수
    n2 = ((N//H)+1 if N%H !=0 else N//H) #호수
    if n2<10:
        print(n1, '0', n2, sep='')
    else:
        print(n1, n2, sep='')