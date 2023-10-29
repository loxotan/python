import sys
input = sys.stdin.readline

N, M, K = map(int, input().split())
portal = [list(map(int, input().split())) for _ in range(K)]
L = int(input())
un_portal = []
if L != 0:
    for _ in range(L):
        un_portal.append(int(input()))

def dfs(C, d):
    global dystopia
    print(C, d, dystopia)
    if C in un_portal:
        print('dystopia')
        dystopia = 0
        exit()
    if d == K:
        if C == 0:
            print('utopia')
            dystopia = 0
            exit()
        else:
            dystopia += 1
            return
            
    D = (C+portal[d][0])%N
    E = (C+portal[d][1])%N
    dfs(D, d+1)
    dfs(E, d+1)

dystopia = 0       
dfs(M, 0)

if dystopia != 0:
    print('dystopia')
        