import sys
input = sys.stdin.readline

N = int(input())
ppl = list(map(int, input().split()))

ppl.sort()

for i in range(1, N):
    ppl[i] = ppl[i-1] + ppl[i]
    
print(sum(ppl))