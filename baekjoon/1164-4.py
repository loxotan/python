l, u = map(int, input().split())
ans = 0
for i in range(l, u)



#2~3 : AA, A-A : 1 + 26
# 0 ,1
#4~6 : A--A, A---A, A----A : 26 + 26^2 + 26^2
# 1, 2, 2
#3~7 : A-A, A--A, A---A, A----A, A-----A : 26, 26, 26^2, 26^2, 26^3
# 1, 1, 2, 2, 3
