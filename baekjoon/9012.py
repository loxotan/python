import sys
input = sys.stdin.readline

n = int(input())

def is_ps(k):
    for _ in range(len(k)//2):
        k = k.replace('()', '')
    return k

for i in range(n):
    p = input().rstrip()
    if len(is_ps(p))==0:
        print('YES')
    else:
        print('NO')