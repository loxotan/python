import sys
input = sys.stdin.readline

channel = input().rstrip()
n = int(input())
gojang = list(map(int, input().split()))
button = list(set(list(range(10))) - set(gojang))

print(button)
