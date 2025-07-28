import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)

def kantor(l):
    if l == 1:
        return('-')
    
    line = kantor(l//3)
    next = line + ' '*(l//3) + line
    return next

while True:
    try:
        n = int(input())
        print(kantor(3**n))
    except:
        break
    