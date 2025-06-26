prime = []
for i in range(2, 100001):
    is_prime = True
    for j in range(2, int(i**0.5) + 1):
        if i % j == 0:
            is_prime = False
            break
    if is_prime:
        prime.append(i)

T = int(input())
for _ in range(T):
    n = int(input())
    for p in prime:
        if p * p > n:  # n이 소수인 경우 처리
            break
        cnt = 0
        while n % p == 0:
            cnt += 1
            n //= p
        if cnt > 0:
            print(p, cnt)
    if n > 1:   # 마지막 남은 n이 1이 아니면 n 자체가 소수
        print(n, 1)