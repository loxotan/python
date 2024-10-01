d, h, w = map(int, input().split())

res = d/((h**2+w**2)**(1/2))
print(int(res*h), int(res*w), sep=' ')