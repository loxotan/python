n = int(input())
jumsu = list(map(int, input().split()))

while n < 5:
    jumsu.append(0)
    n += 1

#국ㅇㅓ, 영어
if jumsu[0] > jumsu[2]:
    a = (jumsu[0]-jumsu[2])*508
else:
    a = (jumsu[2]-jumsu[0])*108

#수학, 탐구
if jumsu[1] > jumsu[3]:
    b = (jumsu[1]-jumsu[3])*212
else:
    b = (jumsu[3]-jumsu[1])*305

#제2외국어
c = jumsu[4]*707


print((a+b+c)*4763)