import sys
input = sys.stdin.readline

n, w, h = map(int, input().split())
for _ in range(n):
    length = int(input())
    if length <= (w**2+h**2)**(1/2):
        print("DA")
    else:
        print("NE")