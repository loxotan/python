import sys
input = sys.stdin.readline

n = int(input())
A = list(map(int, input().split()))
dic = {}
for x in A:
    dic.update({x:1})

m = int(input())
B = list(map(int, input().split()))
for y in B:
    print(dic.get(y, 0))