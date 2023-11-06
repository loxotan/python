str, dex, int, luk, k = map(int, input().split())

v = 4*k - (str+dex+int+luk)
if v > 0:
    print(v)
else:
    print(0)
    