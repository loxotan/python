import sys
input = sys.stdin.readline

lst = [False, False] + [True]*1000100
primes = []
for i in range(1000100):
    if lst[i]:
        primes.append(i)
        for j in range(i*i, 1000100, i):
            lst[j] = False

while True:
    n = int(input())
    if n == 0:
        break

    for i in range(3, n):
        if lst[i]:
            if lst[n-i]:
                a, b = i, n-i
                print(n, "=", a, "+", b, sep = " ")
                break
    else:
        print("Goldbach's conjecture is wrong.")

