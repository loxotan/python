import sys
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    N = int(input())
    if N == 0:
        print(0)
        continue
    clothes = {}
    for __ in range(N):
        name, type_ = input().split()
        if type_ in clothes:
            clothes[type_] += 1
        else:
            clothes[type_] = 1
    ans = 0
    for type_ in clothes:
        if ans == 0:
            ans = clothes[type_] + 1
        else:
            ans *= (clothes[type_] + 1)
    print(ans - 1)
    