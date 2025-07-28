import sys
input = sys.stdin.readline

n = list(map(int, input().split()))
hap = 0
for i in range(5):
    hap += n[i]**2
    
print(hap%10)