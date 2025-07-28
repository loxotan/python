import sys
input = sys.stdin.readline

N = int(input())
mujige = set()
mujige.add('ChongChong')
for _ in range(N):
    a, b = map(str, input().split())
    if a in mujige:
        mujige.add(b)
    if b in mujige:
        mujige.add(a)
    
print(len(mujige))