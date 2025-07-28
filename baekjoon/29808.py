m = int(input())
n = m/4763

# n*4763
# = a*508 + b*212
# or a*108 + b*212
# or a*508 + b*305
# or a*108 + b*305

cnt = 0
lst = []

for a in range(0, 201):
    for b in range(0, 201):
        if a*508 + b*212 == n or a*108 + b*212 == n or a*508 + b*305 == n or a*108 + b*305 == n:
            cnt += 1
            lst.append([a,b])

print(cnt)
if cnt != 0:
    for i in range(cnt):
        print(*lst[i])
