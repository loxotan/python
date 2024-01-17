x1, x2 = map(int, input().split())
a, b, c, d, e = map(int, input().split())

print(int((1/3)*a*(x2**3 - x1**3)+(1/2)*(b-d)*(x2**2-x1**2)+(c-e)*(x2-x1)))