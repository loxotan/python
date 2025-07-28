n, k = map(int, input().split())
jari = [0] + list(map(int, input().split()))
# i 번째 원판이 들어있어야 하는 자리

stk = [[], [], [], []]

for i in range(1, n+1):
    stk[k].append(i)
            

print(stk)
    