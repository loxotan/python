S = int(input())

n = 0
total = 0

while total + (n + 1) <= S:
    n += 1
    total += n

print(n)
