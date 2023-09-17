import sys
input = sys.stdin.readline

while True:
    n = str(input().strip())
    if n == '0':
        break
    else:
        n_rev = n[::-1]
        for i in range(len(n)):
            if n[i] != n_rev[i]:
                print('no')
                break
        else:
            print('yes')
            