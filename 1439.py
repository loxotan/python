S = str(input())

i = 1
n = 0
for i in range(len(S)):
    if S[i] != S[i-1]:
        n += 1
    else:
        continue
print(int(n//2))
