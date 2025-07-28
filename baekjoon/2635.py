n = int(input())
max = -1
max_k = 0
for i in range(1, n+1):
    k = i
    arr = [n, k]
    while k>=0:
        arr.append(arr[-2]-arr[-1])
        k = arr[-2]-arr[-1]
    if len(arr) > max:
        max = len(arr)
        max_k = i

print(max)
max_arr = [n, max_k]
k = max_k
while k>=0:
    max_arr.append(max_arr[-2]-max_arr[-1])
    k = max_arr[-2]-max_arr[-1]
print(*max_arr)