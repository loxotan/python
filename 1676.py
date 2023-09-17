n = int(input())

def fact(n):
    if n == 1:
        return 1
    else:
        answer = 1
        for i in range(2, n+1):
            answer *= i
        return answer
    
m = fact(n)
m = list(map(int, list(str(m))))
cnt = 0
while m[-1] == 0:
    m.pop()
    cnt += 1

print(cnt)