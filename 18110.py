import sys
input = sys.stdin.readline

T = int(input())
jumsu = [int(input()) for _ in range(T)]
jumsu.sort()

cut = round(T*15/100)

jumsu_new = jumsu[cut:T-cut]
if len(jumsu_new) == 0:
    print(0)
else:
    mean = round(sum(jumsu_new)/len(jumsu_new))
    print(mean)