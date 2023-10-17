S = input()

list = [0 for _ in range(26)]
for x in S:
    num = ord(x)-97
    list[num] += 1

print(*list)