A, P = map(int, input().split())

def banbok(i, j):
    jari = []
    next = 0
    while i > 0: # 각 자리 떼내기
        jari.append(i%10)
        i = i//10

    for num in jari: # 각 자리마다 j번 곱해서 더하기
        next += num**j
        
    return next

suyeol = [A]
while True:
    B = banbok(suyeol[-1], P)
    suyeol.append(B)
    if suyeol[-1] in suyeol[:-1]:
        C = suyeol.index(suyeol[-1])
        break
    
print(C)