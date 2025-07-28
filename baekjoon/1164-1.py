N = int(input())
name = input()
K = int(input())

if name == 'annyong':
    if K%2 == 1:
        print(K)
    elif K>1:
        print(K-1)
    else:
        print(K+1)
else:
    if K%2 == 0:
        print(K)
    elif K>1:
        print(K-1)
    else:
        print(K+1)