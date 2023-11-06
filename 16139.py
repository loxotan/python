import sys
input = sys.stdin.readline

X = input().rstrip()
Q = int(input())

finish = {}
for _ in range(Q):
    x, left, right = input().split()
    l = int(left)
    r = int(right)
    
    if finish.get(x) is None:
        S = [0 for i in range(len(X)+1)]
        temp = 0
        for j in range(len(X)):
            if X[j] == x:
                temp += 1
                S[j+1] = temp
            else:
                S[j+1] = temp
        finish[x] = S
        print(S[r+1]-S[l])
        
    else:
        v = finish[x]
        print(v[r+1]-v[l])