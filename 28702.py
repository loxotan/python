import sys
input = sys.stdin.readline

a, b, c = input().rstrip(), input().rstrip(), input().rstrip()

if a.isdigit():
    ans = int(a)+3
elif b.isdigit():
    ans = int(b)+2
elif c.isdigit():
    ans = int(c)+1
    
def fizzbuzz(n):
    if n%3 == 0 and n%5 == 0:
        return "FizzBuzz"
    elif n%3 == 0:
        return "Fizz"
    elif n%5 == 0:
        return "Buzz"
    else:
        return n
    
print(fizzbuzz(ans))