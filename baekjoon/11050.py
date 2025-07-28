n, k = map(int, input().split())

def fact(n):
    if n == 1:
        return 1
    else:
        answer = 1
        for i in range(2, n+1):
            answer *= i
        return answer
    
print(fact(n)//(fact(n-k)*fact(k)))