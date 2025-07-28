import sys
input = sys.stdin.readline

nanjeng = [int(input()) for _ in range(9)]
hap = sum(nanjeng)
nanjeng.sort()

for i in range(9):
    for j in range(i, 9):
        if hap - nanjeng[i] - nanjeng[j] == 100:
            hap = 100
            nanjeng.pop(j)
            nanjeng.pop(i)
            break
    if hap == 100:
        break
        
print(*nanjeng, sep = '\n')