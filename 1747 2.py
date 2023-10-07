n = int(input())
prime = [False]*2 + [True]*1003000

for i in range(2, int(len(prime)**0.5)+1):
    prime[i*2::i] == [False]*len(prime[i*2::i])
    
for i in range(n, len(prime)):
    if prime[i] and str(i) == str(i)[::-1]:
        print(i)
        break
