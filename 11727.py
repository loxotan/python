n = int(input())

# n = n-1 + n-2 * 2

# 1
# 11 2 4
# 111 12 21 14 41
# 1111 112 121 211 114 141 411 22 44 24 42

fib2 = {0:1, 1:1}
for i in range(2, n+1):
    fib2[i] = fib2[i-1] + fib2[i-2]*2
    
print(fib2[n]%10007)