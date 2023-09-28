N = int(input())
a = list(map(int, input().split()))

b = list(set(a))
b.sort()

print(*b)
