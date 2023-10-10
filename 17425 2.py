from sys import stdin
input = stdin.readline

max = 1000000

F = [1] * 1000001
G = [0] * 1000001

def calculate():
    G[1] = G[0] + F[1]
    for i in range(2, max+1):
        for j in range(i, max+1, i):
            F[j] += i
        G[i] = G[i-1] + F[i]
        
calculate()

ans = []

T = int(input())
for _ in range(T):
    ans.append(G[int(input())])

print('\n'.join(map(str, ans)) + '\n')
        