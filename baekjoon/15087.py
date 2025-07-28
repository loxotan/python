Y = input()
X = []
for y in Y:
    X.append(ord(y)-65)
    
#Divide
A, B = X[:(len(X)//2)], X[(len(X)//2):]

#Rotate
AA, BB = [], []
for i in range(len(A)):
    AA.append((A[i]+sum(A))%26)
    BB.append((B[i]+sum(B))%26)

#Merge
ans = []
for i in range(len(A)):
    ans.append((AA[i]+BB[i])%26)

#Decode
res = ''
for x in ans:
    res += str(chr(x+65))
    
print(res)