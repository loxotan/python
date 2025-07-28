import sys
sys.setrecursionlimit(10000)

n = int(input())

good = []
def bkt(a, idx):
    if idx>2:
        for i in range(1, idx//2+1):
            if good[-i:] == good[-2*i:-i]:
                return
            
    if idx == a:
            print(*good, sep='')
            exit(0)  
                      
    for next in range(1, 4):
        good.append(next)
        bkt(a, idx+1)
        good.pop()
        
bkt(n+1, 1)
